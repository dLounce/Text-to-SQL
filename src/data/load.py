import json


def _read(path):
    with open(path) as f:
        return json.load(f)


def load_all(data_root):
    """Load OmniSQL BIRD + Spider train sets.

    OmniSQL rows carry only input_seq/output_seq. source is added here;
    db_id and gold_sql are attached by index from the original dataset
    files, which are index-aligned with the OmniSQL files.
    """
    sources = {
        "bird": ("train_bird.json", "bird/train/train.json", "SQL"),
        "spider": ("train_spider.json", "spider/train_spider.json", "query"),
    }
    rows = []
    for source, (omni_file, src_file, gold_key) in sources.items():
        omni = _read(f"{data_root}/{omni_file}")
        src = _read(f"{data_root}/{src_file}")
        assert len(omni) == len(src), f"{source}: length mismatch"
        for r, s in zip(omni, src):
            assert s["question"].strip() in r["input_seq"], \
                f"{source}: index misalignment"
            r["source"] = source
            r["db_id"] = s["db_id"]
            r["gold_sql"] = s[gold_key]
        rows += omni
    return rows