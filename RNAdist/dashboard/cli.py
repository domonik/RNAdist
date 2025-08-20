import os
import sqlite3
import RNAdist.dashboard
from RNAdist.dashboard.helpers import Database



def cli_wrapper(
        debug: bool = False,
        port: int = 8090,
        host: str = "127.0.0.1",
        processes: int = 1
):


    from RNAdist.dashboard.app import app, get_layout
    db = Database(RNAdist.dashboard.CONFIG["DATABASE"])
    db.create_database(db)
    db.delete_unfinished_jobs()

    app.layout = get_layout()
    app.run(debug=debug, port=port, host=host, processes=processes, threaded=False)


def _cli_wrapper(args):
    cli_wrapper(args.database, debug=False, port=args.port, host=args.host, processes=args.processes)





if __name__ == '__main__':
    database_file = "../../foo_db.db"
    #assert os.path.exists(database_file)
    cli_wrapper(debug=True, processes=3)
