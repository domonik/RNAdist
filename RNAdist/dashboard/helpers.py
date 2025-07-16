import RNA
from types import MappingProxyType
import hashlib
import json
import io
import sqlite3
import numpy as np
import zlib

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
    cursor.execute("SELECT * FROM jobs WHERE user_id = ?", (user_id,))
    jobs = cursor.fetchall()
    conn.close()
    return jobs


def matrix_from_hash(db_path, md_hash):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT matrix FROM submissions WHERE hash = ?", (md_hash,))
    row = cursor.fetchone()
    buf = io.BytesIO(row[0])
    decompressed = zlib.decompress(buf.getvalue())
    matrix = np.load(io.BytesIO(decompressed))
    return matrix

def set_status(db_path, hash_value, status, user_id, header):
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


def insert_submission(sequence, histograms, samples, fc, md, db_path):
    fields, md_hash = hash_model_details(md, sequence)
    non_zero_mask = histograms != 0
    non_zero_any_z = np.any(non_zero_mask, axis=(0, 1))

    # Find the last index where it's True
    last_nonzero_index = np.where(non_zero_any_z)[0].max()
    histograms = histograms[:, :, :last_nonzero_index + 1]
    # Serialize matrix
    buf = io.BytesIO()
    np.save(buf, histograms)
    compressed_blob = zlib.compress(buf.getvalue())

    # Base columns
    submission = {
        "hash": md_hash,
        "sequence": sequence,
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

        values = [(md_hash, struct, count) for struct, count in samples.items()]

        cursor.executemany(
            "INSERT INTO structures (hash, structure, count) VALUES (?, ?, ?)",
            values
        )
    finally:
        conn.commit()
        conn.close()



if __name__ == '__main__':
    md = RNA.md()
    seq = "AAUGCCAUCG"
    f, md_hash = hash_model_details(md, seq)
    print(md_hash)
