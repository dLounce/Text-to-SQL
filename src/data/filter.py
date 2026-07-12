import os
import signal
from multiprocessing import Pool

from tqdm import tqdm

from src.execution.runner import connect_readonly, db_path


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _run_gold(args):
    idx, path, sql, timeout = args
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    con = connect_readonly(path)
    try:
        n = len(con.execute(sql).fetchall())
        return idx, "empty" if n == 0 else "keep"
    except _Timeout:
        return idx, "timeout"
    except Exception:
        return idx, "error"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        con.close()


def execution_filter(rows, data_root, timeout=15, workers=None):
    """Run every gold query and drop rows whose gold errors, times out,
    or returns no rows. Returns (kept_rows, stats)."""
    stats = {"keep": 0, "empty": 0, "timeout": 0, "error": 0, "missing": 0}
    tasks = []
    for i, r in enumerate(rows):
        path = db_path(data_root, r["source"], r["db_id"])
        if os.path.exists(path):
            tasks.append((i, path, r["gold_sql"], timeout))
        else:
            stats["missing"] += 1

    keep_idx = []
    with Pool(workers or os.cpu_count()) as pool:
        for idx, status in tqdm(pool.imap_unordered(_run_gold, tasks,
                                                    chunksize=50),
                                total=len(tasks)):
            stats[status] += 1
            if status == "keep":
                keep_idx.append(idx)

    keep_idx.sort()
    return [rows[i] for i in keep_idx], stats