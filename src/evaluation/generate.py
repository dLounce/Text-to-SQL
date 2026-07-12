import json

from vllm import LLM, SamplingParams


def generate_predictions(model_path, rows, out_path, max_tokens=16000,
                         temperature=0.0, max_model_len=32768):
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=0.92)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate([r["input_seq"] for r in rows], params)

    preds = [
        {
            "idx": i,
            "source": rows[i]["source"],
            "db_id": rows[i]["db_id"],
            "gold_sql": rows[i]["gold_sql"],
            "pred_raw": out.outputs[0].text,
        }
        for i, out in enumerate(outputs)
    ]
    with open(out_path, "w") as f:
        json.dump(preds, f)
    return preds