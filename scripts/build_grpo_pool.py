import argparse
import json
import os
import re
from collections import Counter
from multiprocessing import Pool

from src.execution.runner import connect_readonly, db_path, run_query
from src.execution.scoring import extract_sql, results_match


def generate_rollouts(model_path, rows, out_path, n, temperature, max_tokens,
                      chunk=1000):
    from vllm import LLM, SamplingParams

    done = set()
    if os.path.exists(out_path):
        done = {json.loads(line)["idx"] for line in open(out_path)}

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=32768,
              gpu_memory_utilization=0.92)
    params = SamplingParams(temperature=temperature, n=n,
                            max_tokens=max_tokens, seed=42)

    todo = [i for i in range(len(rows)) if i not in done]
    for start in range(0, len(todo), chunk):
        batch = todo[start:start + chunk]
        outputs = llm.generate([rows[i]["input_seq"] for i in batch], params)
        with open(out_path, "a") as f:
            for i, out in zip(batch, outputs):
                f.write(json.dumps({
                    "idx": i,
                    "source": rows[i]["source"],
                    "db_id": rows[i]["db_id"],
                    "gold_sql": rows[i]["gold_sql"],
                    "gens": [g.text for g in out.outputs],
                }) + "\n")


def _count_correct(args):
    data_root, item, timeout = args
    con = connect_readonly(db_path(data_root, item["source"], item["db_id"]))
    try:
        gold = con.execute(item["gold_sql"]).fetchall()
        ordered = bool(re.search(r"\border\s+by\b", item["gold_sql"], re.I))
        n_ok = 0
        for text in item["gens"]:
            status, pred = run_query(con, extract_sql(text), timeout,
                                     max_rows=len(gold))
            if status == "ok" and results_match(pred, gold, ordered):
                n_ok += 1
        return item["idx"], n_ok
    finally:
        con.close()


def bucket(rollout_path, rows, data_root, out_path, timeout=15):
    items = [json.loads(line) for line in open(rollout_path)]
    tasks = [(data_root, item, timeout) for item in items]
    with Pool(os.cpu_count()) as pool:
        counts = dict(pool.imap_unordered(_count_correct, tasks, chunksize=5))

    dist = Counter(counts.values())
    print("n_correct distribution", dict(sorted(dist.items())))

    pool_rows = [{**rows[i], "n_correct": n} for i, n in counts.items()]
    with open(out_path, "w") as f:
        json.dump(pool_rows, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--rollouts", default="outputs/data/rollouts.jsonl")
    parser.add_argument("--out", default="outputs/data/grpo_pool.json")
    parser.add_argument("--stage", choices=["generate", "bucket"], required=True)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16000)
    args = parser.parse_args()

    rows = json.load(open(args.train_file))
    if args.stage == "generate":
        generate_rollouts(args.model, rows, args.rollouts,
                          args.num_generations, args.temperature,
                          args.max_tokens)
    else:
        bucket(args.rollouts, rows, args.data_root, args.out)


if __name__ == "__main__":
    main()