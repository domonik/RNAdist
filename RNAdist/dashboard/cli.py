import os
import sqlite3
import RNAdist.dashboard
from RNAdist.dashboard.helpers import get_md_fields, create_database



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
        create_database(db)
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
