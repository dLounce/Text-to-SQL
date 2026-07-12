import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

from src.execution.scoring import score_prediction


def _score(args):
    idx, data_root, source, db_id, gold_sql, pred_raw, timeout = args
    return idx, score_prediction(data_root, source, db_id, gold_sql, pred_raw, timeout)


def evaluate_predictions(preds, data_root, out_path=None, timeout=15, workers=None):
    """Execution accuracy overall and per source. Returns (results, report)."""
    tasks = [
        (p["idx"], data_root, p["source"], p["db_id"], p["gold_sql"], p["pred_raw"], timeout)
        for p in preds
    ]
    with Pool(workers or os.cpu_count()) as pool:
        results = dict(pool.imap_unordered(_score, tasks, chunksize=20))

    overall = Counter(results.values())
    per_source = defaultdict(Counter)
    for p in preds:
        per_source[p["source"]][results[p["idx"]]] += 1

    report = {
        "overall": {"accuracy": overall["correct"] / len(preds), "counts": dict(overall)},
        "per_source": {
            s: {"accuracy": c["correct"] / sum(c.values()), "counts": dict(c)}
            for s, c in per_source.items()
        },
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump({"results": {str(k): v for k, v in results.items()},
                       "report": report}, f, indent=2)
    return results, report