import random
from collections import defaultdict


def split_by_db(rows, test_frac=0.10, val_frac=0.05, seed=42):
    """Split at the database level, stratified by source, quotas by row
    count. Greedily fills test, then val; the rest is train. No db_id
    appears in more than one split."""
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for source in ("bird", "spider"):
        src_rows = [r for r in rows if r["source"] == source]
        by_db = defaultdict(list)
        for r in src_rows:
            by_db[r["db_id"]].append(r)
        dbs = list(by_db)
        rng.shuffle(dbs)

        total = len(src_rows)
        quota = {"test": test_frac * total, "val": val_frac * total}
        filled = {"test": 0, "val": 0}
        for db in dbs:
            if filled["test"] < quota["test"]:
                splits["test"] += by_db[db]
                filled["test"] += len(by_db[db])
            elif filled["val"] < quota["val"]:
                splits["val"] += by_db[db]
                filled["val"] += len(by_db[db])
            else:
                splits["train"] += by_db[db]

    key = lambda part: {(r["source"], r["db_id"]) for r in part}
    assert not key(splits["train"]) & key(splits["test"])
    assert not key(splits["train"]) & key(splits["val"])
    assert not key(splits["val"]) & key(splits["test"])
    return splits