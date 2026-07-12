import sqlite3
import time


def db_path(data_root, source, db_id):
    paths = {
        "bird": f"{data_root}/bird/train/train_databases/{db_id}/{db_id}.sqlite",
        "spider": f"{data_root}/spider/database/{db_id}/{db_id}.sqlite",
    }
    return paths[source]


def connect_readonly(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA query_only=ON")
    return con


def run_query(con, sql, timeout=15, max_rows=None):
    """Execute sql with a wall-clock limit enforced via sqlite progress handler.

    Returns (status, rows) where status is one of ok, timeout, error, toobig.
    """
    timed_out = [False]
    start = time.time()

    def guard():
        if time.time() - start > timeout:
            timed_out[0] = True
            return 1
        return 0

    con.set_progress_handler(guard, 100_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchall() if max_rows is None else cur.fetchmany(max_rows + 1)
        if max_rows is not None and len(rows) > max_rows:
            return "toobig", None
        return "ok", rows
    except Exception:
        return ("timeout" if timed_out[0] else "error"), None
    finally:
        con.set_progress_handler(None, 0)