import argparse
import json
import os
import random
from src.data.load import load_all
from src.data.filter import execution_filter
from src.data.split import split_by_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="outputs/data")
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_all(args.data_root)
    print(f"loaded {len(rows)} rows")

    kept, stats = execution_filter(rows, args.data_root, args.timeout)
    print(f"filter kept {len(kept)} {stats}")

    splits = split_by_db(kept)
    for name, split_rows in splits.items():
        with open(f"{args.out_dir}/{name}.json", "w") as f:
            json.dump(split_rows, f)
        print(f"{name}: {len(split_rows)}")

    for name in ("train", "val"):
        sft = [{"prompt": r["input_seq"], "completion": r["output_seq"]}
               for r in splits[name]]
        with open(f"{args.out_dir}/sft_{name}.json", "w") as f:
            json.dump(sft, f)

    random.seed(42)
    val = splits["val"]
    by_src = {s: [r for r in val if r["source"] == s] for s in ("bird", "spider")}
    n_bird = round(150 * len(by_src["bird"]) / len(val))
    dev150 = random.sample(by_src["bird"], n_bird) + \
        random.sample(by_src["spider"], 150 - n_bird)
    with open(f"{args.out_dir}/grpo_dev150.json", "w") as f:
        json.dump(dev150, f)   



if __name__ == "__main__":
    main()