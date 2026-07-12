import re
from collections import Counter

from src.execution.runner import connect_readonly, db_path, run_query


def extract_sql(text):
    blocks = re.findall(r"```sql\s*(.*?)```", text, re.S | re.I)
    if blocks:
        return blocks[-1].strip()
    m = re.search(r"```sql\s*(.*)", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    return text.strip()


def results_match(pred_rows, gold_rows, ordered):
    if ordered:
        return [tuple(r) for r in pred_rows] == [tuple(r) for r in gold_rows]
    return Counter(map(tuple, pred_rows)) == Counter(map(tuple, gold_rows))


def score_prediction(data_root, source, db_id, gold_sql, pred_text, timeout=15):
    """Execution-match a raw model output against gold. Returns a status string."""
    con = connect_readonly(db_path(data_root, source, db_id))
    try:
        gold = con.execute(gold_sql).fetchall()
        status, pred = run_query(con, extract_sql(pred_text), timeout, max_rows=len(gold))
        if status == "toobig":
            return "wrong"
        if status != "ok":
            return status
        ordered = bool(re.search(r"\border\s+by\b", gold_sql, re.I))
        return "correct" if results_match(pred, gold, ordered) else "wrong"
    finally:
        con.close()