import os
import sqlite3
import RNAdist.dashboard
from RNAdist.dashboard.helpers import get_md_fields

def sqlite_type(val):
    if isinstance(val, int):
        return "INTEGER"
    elif isinstance(val, float):
        return "FLOAT"
    elif isinstance(val, str):
        return "TEXT"
    else:
        return "BLOB"  # fallback for arrays, None, etc.



def cli_wrapper(
        db: str,
        debug: bool = False,
        port: int = 8090,
        host: str = "127.0.0.1",
        processes: int = 1
):
    RNAdist.dashboard.DATABASE_FILE = db

    from RNAdist.dashboard.app import app, get_layout
    if not os.path.exists(db):
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        fields = get_md_fields()
        md_columns = [
            f"{key} {sqlite_type(fields[key])} NOT NULL"
            for key in fields.keys()
        ]
        table_sql = f"""
CREATE TABLE submissions (
    hash BLOB PRIMARY KEY,
    sequence TEXT NOT NULL,
    length INTEGER NOT NULL,
    matrix BLOB,
    {',\n    '.join(md_columns)},
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
        jobs = f"""
CREATE TABLE jobs (
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

    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM jobs WHERE status != 'finished';")
    conn.commit()
    conn.close()

    app.layout = get_layout()
    app.run(debug=debug, port=port, host=host, processes=processes, threaded=False)


def _cli_wrapper(args):
    cli_wrapper(args.config, args.run_dir, args.debug, args.port, args.host, args.processes)





if __name__ == '__main__':
    database_file = "mydata.db"
    cli_wrapper(db=database_file, debug=True, processes=3)
