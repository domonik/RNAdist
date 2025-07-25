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

db_lock = Lock()
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


def check_user_header_combination(db_path, user_id, header):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM jobs WHERE user_id = ? AND header = ? LIMIT 1", (user_id, header))
    row = cursor.fetchone()
    return bool(row)


def check_user_hash_combination(db_path, user_id, md_hash):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT header FROM jobs WHERE user_id = ? AND hash = ? LIMIT 1", (user_id, md_hash))
    row = cursor.fetchone()
    return row


def _delete_old_entries(db_path):
    conn = sqlite3.connect(db_path)  # replace with your DB path
    cursor = conn.cursor()
    with db_lock:
        cursor.execute("""
        UPDATE jobs
        SET status = 'deleted'
        WHERE hash IN (
            SELECT hash FROM submissions
            WHERE created_at < datetime('now', '-7 days')
            AND protected = 0
        );
        """)
        cursor.execute("""
            DELETE FROM submissions
            WHERE created_at < datetime('now', '-7 days')
            AND protected = 0
                       """)
        conn.commit()
    conn.close()


def _cleanup_oldest_if_needed(db_path):
    with db_lock:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Count entries
        cursor.execute("SELECT COUNT(*) FROM submissions")
        count = cursor.fetchone()[0]

        if count >= 1000:
            print(f"[cleanup] Table has {count} entries. Deleting 10 oldest.")

            cursor.execute("""
                           SELECT hash FROM submissions
                           WHERE protected = 0
                           ORDER BY created_at ASC
                               LIMIT 10
                           """)
            hashes_to_delete = [row[0] for row in cursor.fetchall()]

            if hashes_to_delete:
                placeholders = ",".join("?" for _ in hashes_to_delete)

                # Step 2: Update jobs table
                cursor.execute(f"""
                        UPDATE jobs
                        SET status = 'deleted'
                        WHERE hash IN ({placeholders})
                    """, hashes_to_delete)

                # Step 3: Delete from submissions
                cursor.execute(f"""
                        DELETE FROM submissions
                        WHERE hash IN ({placeholders})
                    """, hashes_to_delete)

                conn.commit()
        else:
            print(f"[cleanup] Table size ({count}) below threshold. No cleanup.")

        conn.close()

def database_cleanup(db_path):
    _delete_old_entries(db_path)
    _cleanup_oldest_if_needed(db_path)



def check_hash_exists(db_path, hash_value):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    # Check if hash exists
    cursor.execute("SELECT status FROM jobs WHERE hash = ?", (hash_value,))
    row = cursor.fetchone()
    if row is not None:
        status = row[0]
        return status
    else:
        return False


def get_jobs_of_user(db_path, user_id):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT *
                   FROM jobs
                   LEFT JOIN submissions ON jobs.hash = submissions.hash
                   WHERE jobs.user_id = ?
                   """, (user_id,))
    jobs = cursor.fetchall()
    conn.close()
    return jobs

@lru_cache(maxsize=8) # This only works in production with gunicorn
def matrix_from_hash(db_path, md_hash, return_mfe: bool = False):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("SELECT matrix, length, mfe FROM submissions WHERE hash = ?", (md_hash,))
    row = cursor.fetchone()
    buf = io.BytesIO(row["matrix"])
    decompressed = zlib.decompress(buf.getvalue())
    compressed_mat = np.load(io.BytesIO(decompressed))
    z = compressed_mat.shape[1]
    n = row["length"]
    tri_upper = np.triu_indices(n)
    matrix = np.zeros((n, n, z), dtype=compressed_mat.dtype)
    matrix[tri_upper[0][:, None], tri_upper[1][:, None], np.arange(z)] = compressed_mat
    matrix[tri_upper[1][:, None], tri_upper[0][:, None], np.arange(z)] = compressed_mat  # mirror

    conn.close()
    if return_mfe:
        s = time.time()
        mfe_distances = distances_from_structure(row["mfe"])
        e = time.time()
        print(e - s, "seconds")

        return matrix, mfe_distances
    return matrix

def set_status(db_path, hash_value, status, user_id, header):
    with db_lock:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO jobs (hash, status, user_id, header)
            VALUES (?, ?, ?, ?)
        """, (hash_value, status, user_id, header))
        conn.commit()
        conn.close()


def get_structures_and_length_for_hash(db_path, md_hash):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM structures WHERE hash = ? ORDER BY count DESC", (md_hash,))
    rows = cursor.fetchall()
    cursor.execute("SELECT length FROM submissions WHERE hash = ? LIMIT 1", (md_hash,))
    length = cursor.fetchone()["length"]
    conn.close()
    return rows, length


def get_structures_by_ids(db_path, structure_indices):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    placeholders = ','.join(['?'] * len(structure_indices))  # "?,?,?" if 3 hashes

    query = f"SELECT * FROM structures WHERE id IN ({placeholders})"
    cursor.execute(query, structure_indices)
    rows = cursor.fetchall()
    return rows


def insert_submission(sequence, histograms, structure_cache, fc, md, db_path):
    fields, md_hash = hash_model_details(md, sequence)
    non_zero_mask = histograms != 0
    non_zero_any_z = np.any(non_zero_mask, axis=(0, 1))

    # Find the last index where it's True
    last_nonzero_index = np.where(non_zero_any_z)[0].max()
    histograms = histograms[:, :, :last_nonzero_index + 1]


    n, _, z = histograms.shape

# Get indices of upper triangle
    tri_upper_indices = np.triu_indices(n)
    histograms = histograms[tri_upper_indices[0][:, None], tri_upper_indices[1][:, None], np.arange(z)]    # Serialize matrix
    buf = io.BytesIO()
    np.save(buf, histograms)
    compressed_blob = zlib.compress(buf.getvalue())

    # Base columns
    submission = {
        "hash": md_hash,
        "sequence": sequence,
        "mfe": fc.mfe()[0],
        "matrix": compressed_blob,
        "length": fc.length,
    }

    # Merge in model detail fields
    submission.update(fields)

    # Ensure order is correct for SQLite
    columns = ", ".join(submission.keys())
    placeholders = ", ".join("?" for _ in submission)
    values = tuple(submission.values())

    # Connect and insert
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            INSERT INTO submissions ({columns}) VALUES ({placeholders})
        """, values)

        values = [(md_hash, struct, count) for struct, count in structure_cache.items()]

        cursor.executemany(
            "INSERT INTO structures (hash, structure, count) VALUES (?, ?, ?)",
            values
        )
    finally:
        conn.commit()
        conn.close()


def sqlite_type(val):
    if isinstance(val, int):
        return "INTEGER"
    elif isinstance(val, float):
        return "FLOAT"
    elif isinstance(val, str):
        return "TEXT"
    else:
        return "BLOB"  # fallback for arrays, None, etc.




def create_database(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    fields = get_md_fields()
    md_columns = [
        f"{key} {sqlite_type(fields[key])} NOT NULL"
        for key in fields.keys()
    ]
    table_sql = f"""
CREATE TABLE IF NOT EXISTS submissions (
    hash BLOB PRIMARY KEY,
    sequence TEXT NOT NULL,
    mfe TEXT NOT NULL,
    length INTEGER NOT NULL,
    matrix BLOB,
    protected BOOLEAN DEFAULT FALSE NOT NULL, 
    {',\n    '.join(md_columns)},
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
    jobs = f"""
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash BLOB NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    header TEXT NOT NULL,
    UNIQUE (hash, user_id),
    UNIQUE (user_id, header),
    FOREIGN KEY (hash) REFERENCES submissions(hash)
    );
"""
    cursor.execute(jobs)
    cursor.execute(table_sql)
    cursor.execute("""
CREATE TABLE IF NOT EXISTS structures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash BLOB NOT NULL,
    structure BLOB NOT NULL,
    count INTEGER NOT NULL,
    FOREIGN KEY (hash) REFERENCES submissions(hash)
);
""")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON submissions(hash);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_structure_id ON structures(hash);")
    conn.commit()
    conn.close()




if __name__ == '__main__':
    md = RNA.md()
    seq = "AAUGCCAUCG"
    f, md_hash = hash_model_details(md, seq)
    print(md_hash)
