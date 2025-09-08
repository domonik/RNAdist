from functools import lru_cache

import RNA
from types import MappingProxyType
import hashlib
import json
import io
import sqlite3
import numpy as np
import zlib
from multiprocessing import Lock
from RNAdist.sampling.cpp.sampling import distances_from_structure
import time
import logging
import datetime
from sqlalchemy import create_engine, text
from collections import OrderedDict
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def get_md_fields():
    fields, _ = hash_model_details(RNA.md(), "A")
    return fields

def hash_model_details(md: RNA.md, sequence):
    fields = {
        key: getattr(md, key)
        for key in dir(md)
        if not key.startswith("_") and not callable(getattr(md, key)) and not key in ["alias", "this", "pair", "rtype", "circ_alpha0", "nonstandards", "thisown"]
    }

    payload = {
        "sequence": sequence,
        "model_details": {k: fields[k] for k in sorted(fields)}
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    md_hash = hashlib.sha256(encoded).digest()
    return fields, md_hash

class Database:
    def __init__(self, config, max_cache_size:int =8):
        self.engine = self._get_engine(config)
        self._matrix_cache = {}
        self.max_cache_size = max_cache_size
        self.config = config


    def delete_unfinished_jobs(self):
        with self.engine.connect() as conn:
            conn.execute(
                text("DELETE FROM jobs WHERE status != 'finished';")
            )
            conn.commit()

    def _get_engine(self, config):
        if config["type"] == "sqlite":
            return create_engine(
                f"sqlite:///{config['path']}",
                connect_args={"check_same_thread": False}
            )
        elif config["type"] == "postgresql":
            return create_engine(
                f"postgresql+psycopg2://{config['user']}:{config['password']}@"
                f"{config['host']}/{config['database']}"
            )
        else:
            raise ValueError(f"Unsupported db_type: {config['db_type']}")

    # ---------- Queries ----------

    def check_user_header_combination(self, user_id, header):
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT status FROM jobs WHERE user_id = :user_id AND header = :header LIMIT 1"),
                {"user_id": user_id, "header": header}
            ).fetchone()
            return bool(result)

    def check_user_hash_combination(self, user_id, md_hash):
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT header FROM jobs WHERE user_id = :user_id AND hash = :md_hash LIMIT 1"),
                {"user_id": user_id, "md_hash": md_hash}
            ).fetchone()
            return result[0] if result else None

    def check_hash_exists(self, hash_value):
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT status FROM jobs WHERE hash = :hash_value LIMIT 1"),
                {"hash_value": hash_value}
            ).fetchone()
            return result[0] if result else False

    # ---------- Maintenance ----------

    def _delete_old_entries(self):
        cutoff_date = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7)
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                     DELETE FROM jobs
                     WHERE hash IN (
                         SELECT hash FROM submissions
                         WHERE created_at < :cutoff
                           AND protected = FALSE
                     )
                     """),
                {"cutoff": cutoff_date}
            )
            conn.execute(
                text("""
                     DELETE FROM submissions
                     WHERE created_at < :cutoff
                       AND protected = FALSE
                     """),
                {"cutoff": cutoff_date}
            )

    def _cleanup_oldest_if_needed(self, max_entries: int = 1000, batch_size: int = 10):
        with self.engine.begin() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM submissions")).scalar()

            if count >= max_entries:
                logger.info("[cleanup] Table has %s entries. Deleting %s oldest.", count, batch_size)

                result = conn.execute(
                    text("""
                         SELECT hash FROM submissions
                         WHERE protected = 0
                         ORDER BY created_at ASC
                             LIMIT :limit
                         """),
                    {"limit": batch_size}
                )
                hashes_to_delete = [row[0] for row in result.fetchall()]

                if hashes_to_delete:
                    placeholders = ", ".join(f":h{i}" for i in range(len(hashes_to_delete)))
                    params = {f"h{i}": h for i, h in enumerate(hashes_to_delete)}

                    conn.execute(
                        text(f"DELETE FROM jobs WHERE hash IN ({placeholders})"),
                        params
                    )
                    conn.execute(
                        text(f"DELETE FROM submissions WHERE hash IN ({placeholders})"),
                        params
                    )
            else:
                logger.info("[cleanup] Table size (%s) below threshold. No cleanup.", count)

    def database_cleanup(self):
        self._delete_old_entries()
        self._cleanup_oldest_if_needed()

    def get_submission_for_user_header(self, user_id, header):
        """Return submission details for a given user and header."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                     SELECT submissions.length,
                            submissions.hash,
                            submissions.sequence,
                            submissions.mfe
                     FROM jobs
                              JOIN submissions ON jobs.hash = submissions.hash
                     WHERE jobs.user_id = :user_id AND jobs.header = :header
                     """),
                {"user_id": user_id, "header": header}
            ).mappings().first()  # returns a dict-like row or None
        return result

    def get_finished_headers(self, user_id):
        """Return distinct headers for finished jobs of a given user."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                     SELECT DISTINCT header
                     FROM jobs
                     WHERE status = :status AND user_id = :user_id
                     """),
                {"status": "finished", "user_id": user_id}
            )
            headers = [row["header"] for row in result.mappings().all()]
        return headers


    def get_jobs_of_user(self, user_id):
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                     SELECT jobs.hash AS job_hash,
                            submissions.hash AS submission_hash,
                            jobs.user_id,
                            jobs.status,
                            jobs.header,
                            submissions.sequence,
                            submissions.mfe,
                            submissions.length,
                            submissions.temperature,
                            submissions.max_bp_span
                     FROM jobs
                              LEFT JOIN submissions ON jobs.hash = submissions.hash
                     WHERE jobs.user_id = :user_id
                     """),
                {"user_id": user_id}
            )
        return result.mappings().all()

    def _fetch_matrix_from_db(self, md_hash, return_mfe: bool = False):
        """Fetch the matrix (and optional MFE distances) from the database."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT matrix, length, mfe FROM submissions WHERE hash = :md_hash"),
                {"md_hash": md_hash}
            ).mappings().first()

        buf = io.BytesIO(row["matrix"])
        decompressed = zlib.decompress(buf.getvalue())
        compressed_mat = np.load(io.BytesIO(decompressed))
        z = compressed_mat.shape[1]
        n = row["length"]

        tri_upper = np.triu_indices(n)
        matrix = np.zeros((n, n, z), dtype=compressed_mat.dtype)
        matrix[tri_upper[0][:, None], tri_upper[1][:, None], np.arange(z)] = compressed_mat
        matrix[tri_upper[1][:, None], tri_upper[0][:, None], np.arange(z)] = compressed_mat  # mirror

        if return_mfe:
            mfe_distances = distances_from_structure(row["mfe"])
            return matrix, mfe_distances

        return matrix

    def matrix_from_hash(self, md_hash, return_mfe: bool = False):
        """Return matrix with optional caching."""
        s = time.time()

        key = (md_hash, return_mfe)
        if key in self._matrix_cache:
            return self._matrix_cache[key]

        matrix = self._fetch_matrix_from_db(md_hash, return_mfe)
        self._matrix_cache[key] = matrix
        if len(self._matrix_cache) > self.max_cache_size:
            # Evict oldest
            self._matrix_cache.popitem(last=False)
        e = time.time()
        logger.info(f"fetching matrix took {e-s} seconds")
        return matrix

    def set_status(self, hash_value, status, user_id, header):
        """Insert or update a job's status."""
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                     INSERT INTO jobs (hash, status, user_id, header)
                     VALUES (:hash, :status, :user_id, :header)
                         ON CONFLICT(hash, user_id) DO UPDATE SET
                         status = excluded.status,
                         header = excluded.header
                     """),
                {"hash": hash_value, "status": status, "user_id": user_id, "header": header}
            )


    def get_structures_and_length_for_hash(self, md_hash):
        """Return all structures for a hash (ordered by count) and the sequence length."""
        with self.engine.connect() as conn:
            # Fetch structures
            structures = conn.execute(
                text("SELECT * FROM structures WHERE hash = :md_hash ORDER BY num_samples DESC"),
                {"md_hash": md_hash}
            ).mappings().all()  # returns list of dict-like rows

            # Fetch length
            length_row = conn.execute(
                text("SELECT length FROM submissions WHERE hash = :md_hash LIMIT 1"),
                {"md_hash": md_hash}
            ).mappings().first()
            length = length_row["length"] if length_row else None

        return structures, length

    def get_structures_by_ids(self, structure_indices):
        """Fetch structures given a list of structure IDs."""
        if not structure_indices:
            return []

        placeholders = ", ".join(f":id{i}" for i in range(len(structure_indices)))
        params = {f"id{i}": sid for i, sid in enumerate(structure_indices)}

        with self.engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM structures WHERE id IN ({placeholders})"),
                params
            ).mappings().all()  # dict-like rows

        return rows

    @staticmethod
    def compress_histograms(histograms: np.ndarray) -> bytes:
        """Trim histograms to last non-zero z-slice and compress them."""
        non_zero_mask = histograms != 0
        non_zero_any_z = np.any(non_zero_mask, axis=(0, 1))
        last_nonzero_index = np.where(non_zero_any_z)[0].max()
        histograms = histograms[:, :, :last_nonzero_index + 1]

        n, _, z = histograms.shape
        tri_upper_indices = np.triu_indices(n)
        histograms = histograms[tri_upper_indices[0][:, None], tri_upper_indices[1][:, None], np.arange(z)]

        buf = io.BytesIO()
        np.save(buf, histograms)
        compressed_blob = zlib.compress(buf.getvalue())
        return compressed_blob

    @staticmethod
    def build_submission_dict(sequence, fc, md, compressed_blob, protected=False):
        """Build the submission dictionary including model detail fields."""
        fields, md_hash = hash_model_details(md, sequence)
        submission = {
            "hash": md_hash,
            "sequence": sequence,
            "mfe": fc.mfe()[0],
            "matrix": compressed_blob,
            "length": fc.length,
            "protected": protected,
        }
        submission.update(fields)
        return submission, md_hash

    def insert_submission_rows(self, submission: dict, structure_cache: dict = None):
        """Insert the submission and related structures into the DB."""
        with self.engine.begin() as conn:
            cols = ", ".join(submission.keys())
            params = ", ".join(f":{k}" for k in submission.keys())

            # Prefix existing columns with table name to avoid ambiguity
            update_clause = ", ".join(
                f"{k} = COALESCE(submissions.{k}, EXCLUDED.{k})"
                for k in submission.keys() if k != "hash"
            )

            conn.execute(
                text(f"""
                    INSERT INTO submissions ({cols}) VALUES ({params})
                    ON CONFLICT (hash) DO UPDATE
                    SET {update_clause}
                """),
                submission
            )

            if structure_cache is not None:
                    # Insert structures
                    structure_rows = [{"hash": submission["hash"], "structure": s, "num_samples": c}
                                      for s, c in structure_cache.items()]
                    if structure_rows:
                        conn.execute(
                            text("""
                                 INSERT INTO structures (hash, structure, num_samples)
                                 VALUES (:hash, :structure, :num_samples)
                                     ON CONFLICT (hash, structure) DO UPDATE
                                                                          SET num_samples = EXCLUDED.num_samples
                                 """),
                            structure_rows
                        )


    def compress_and_insert_into_submissions(self, sequence, histograms, structure_cache, fc, md, protected=False):
        """High-level function to insert a submission and its structures."""
        compressed_blob = self.compress_histograms(histograms)
        submission, md_hash = self.build_submission_dict(sequence, fc, md, compressed_blob, protected)
        self.insert_submission_rows(submission, structure_cache)
        return md_hash

    def create_database(self, create_file_if_missing=True):
        """Create the SQLite database file and tables if they don't exist."""

    # Ensure the SQLite file exists
        if self.config["type"] == "sqlite":
            # Ensure SQLite file exists
            if create_file_if_missing and not os.path.exists(self.config["path"]):
                Path(self.config["path"]).touch()
            blob_type = "BLOB"
            auto_inc = "INTEGER PRIMARY KEY AUTOINCREMENT"
            timestamp_default = "CURRENT_TIMESTAMP"
        elif self.config["type"] == "postgresql":
            blob_type = "BYTEA"
            auto_inc = "SERIAL PRIMARY KEY"
            timestamp_default = "NOW()"
        else:
            raise ValueError(f"Unsupported database type: {self.config['type']}")

        fields = get_md_fields()
        md_columns = [
            f"{key} {sqlite_type(fields[key])}"
            for key in fields.keys()
        ]

        submissions_sql = f"""
        CREATE TABLE IF NOT EXISTS submissions (
            hash {blob_type} PRIMARY KEY,
            sequence TEXT NOT NULL,
            mfe TEXT,
            length INTEGER NOT NULL,
            matrix {blob_type},
            protected BOOLEAN DEFAULT FALSE NOT NULL, 
            {',\n    '.join(md_columns)},
            created_at TIMESTAMP DEFAULT {timestamp_default}
        );
        """

        jobs_sql = f"""
        CREATE TABLE IF NOT EXISTS jobs (
            id {auto_inc},
            hash {blob_type} NOT NULL,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL,
            header TEXT NOT NULL,
            UNIQUE (hash, user_id),
            UNIQUE (user_id, header),
            FOREIGN KEY (hash) REFERENCES submissions(hash)
        );
        """

        structures_sql = f"""
        CREATE TABLE IF NOT EXISTS structures (
            id {auto_inc},
            hash {blob_type} NOT NULL,
            structure {blob_type} NOT NULL,
            num_samples INTEGER NOT NULL,
            FOREIGN KEY (hash) REFERENCES submissions(hash),
            UNIQUE(hash, structure)
        );
        """

        index_sql = [
            "CREATE INDEX IF NOT EXISTS idx_hash ON submissions(hash);",
            "CREATE INDEX IF NOT EXISTS idx_structure_id ON structures(hash);"
        ]

        with self.engine.begin() as conn:
            conn.execute(text(submissions_sql))
            conn.execute(text(jobs_sql))
            conn.execute(text(structures_sql))
            for idx in index_sql:
                conn.execute(text(idx))

def sqlite_type(val):
    if isinstance(val, int):
        return "INTEGER"
    elif isinstance(val, float):
        return "FLOAT"
    elif isinstance(val, str):
        return "TEXT"
    else:
        return "BLOB"  # fallback for arrays, None, etc.




if __name__ == '__main__':
    md = RNA.md()
    seq = "AAUGCCAUCG"
    f, md_hash = hash_model_details(md, seq)
    print(md_hash)
