import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from src.execution.runner import connect_readonly, db_path, run_query
from src.execution.scoring import extract_sql, results_match

REWARD_CORRECT = 1.0
REWARD_EXECUTABLE = 0.1
REWARD_INVALID = 0.0


class SqlReward:
    def __init__(self, data_root, timeout=15, workers=32):
        self.data_root = data_root
        self.timeout = timeout
        self.pool = ThreadPoolExecutor(workers)
        self.gold_cache = {}

    def gold_rows(self, source, db_id, gold_sql):
        key = (source, db_id, gold_sql)
        if key in self.gold_cache:
            return self.gold_cache[key]
        con = connect_readonly(db_path(self.data_root, source, db_id))
        try:
            rows = con.execute(gold_sql).fetchall()
        finally:
            con.close()
        if len(rows) < 50_000:
            self.gold_cache[key] = rows
        return rows

    def score_one(self, args):
        text, source, db_id, gold_sql = args
        try:
            gold = self.gold_rows(source, db_id, gold_sql)
        except Exception:
            print(f"[goldfail] {source}/{db_id}")
            return REWARD_INVALID
        con = connect_readonly(db_path(self.data_root, source, db_id))
        try:
            status, pred = run_query(con, extract_sql(text), self.timeout,
                                     max_rows=len(gold))
            if status == "error":
                return REWARD_INVALID
            if status in ("timeout", "toobig"):
                return REWARD_EXECUTABLE
            ordered = bool(re.search(r"\border\s+by\b", gold_sql, re.I))
            return REWARD_CORRECT if results_match(pred, gold, ordered) \
                else REWARD_EXECUTABLE
        finally:
            con.close()

    def __call__(self, prompts, completions, gold_sql, source, db_id, **kwargs):
        t0 = time.time()
        out = list(self.pool.map(self.score_one,
                                 zip(completions, source, db_id, gold_sql)))
        print(f"[reward] {time.time() - t0:.1f}s")
        return out


class DevEvalCallback(TrainerCallback):
    def __init__(self, reward, dev_rows, cfg):
        self.reward = reward
        self.dev_rows = dev_rows
        self.every = cfg["dev_eval_steps"]
        self.patience = cfg["patience"]
        self.out_dir = cfg["output_dir"]
        self.max_new = cfg["max_completion_length"]
        self.step_offset = cfg.get("step_offset", 0)
        self.trainer = None  # set after GRPOTrainer is constructed

        self.hist_path = os.path.join(self.out_dir, "dev_history.json")
        self.history = json.load(open(self.hist_path)) \
            if os.path.exists(self.hist_path) else []
        self.best = max((h["acc"] for h in self.history), default=-1.0)
        self.flat = 0

    def evaluate(self):
        outs = self.trainer.vllm_generation.llm.generate(
            [r["input_seq"] for r in self.dev_rows],
            SamplingParams(temperature=0.0, max_tokens=self.max_new))
        args = [(o.outputs[0].text, r["source"], r["db_id"], r["gold_sql"])
                for o, r in zip(outs, self.dev_rows)]
        scores = list(self.reward.pool.map(self.reward.score_one, args))
        return sum(s == REWARD_CORRECT for s in scores) / len(scores)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every:
            return
        acc = self.evaluate()
        step = state.global_step + self.step_offset
        self.history.append({"step": step, "acc": acc})
        with open(self.hist_path, "w") as f:
            json.dump(self.history, f)
        self.trainer.model.save_pretrained(os.path.join(self.out_dir, "lora_last"))
        if acc > self.best:
            self.best, self.flat = acc, 0
            self.trainer.model.save_pretrained(os.path.join(self.out_dir, "lora_best"))
        else:
            self.flat += 1
            if self.flat >= self.patience:
                control.should_training_stop = True
        print(f"[dev150] step {step} | acc {acc:.3f} "
              f"| best {self.best:.3f} | flat {self.flat}")


def build_dataset(pool_file, tokenizer, max_prompt_length, seed):
    rows = json.load(open(pool_file))
    rows = [r for r in rows if 0 < r["n_correct"] < 8]
    rows = [r for r in rows
            if len(tokenizer(r["input_seq"]).input_ids) <= max_prompt_length]
    return Dataset.from_list([
        {"prompt": r["input_seq"], "gold_sql": r["gold_sql"],
         "source": r["source"], "db_id": r["db_id"]} for r in rows
    ]).shuffle(seed=seed)


def load_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(
        base, cfg["init_adapter"], is_trainable=True)
    return model, tokenizer


def train(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    model, tokenizer = load_model(cfg)

    dataset = build_dataset(cfg["pool_file"], tokenizer,
                            cfg["max_prompt_length"], cfg["seed"])
    print("train prompts:", len(dataset))
    dev_rows = json.load(open(cfg["dev_file"]))
    reward = SqlReward(cfg["data_root"], cfg["timeout"])

    args = GRPOConfig(
        output_dir=os.path.join(cfg["output_dir"], "ckpt"),
        learning_rate=cfg["lr"],
        lr_scheduler_type="constant",
        warmup_steps=cfg["warmup_steps"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_generations=cfg["num_generations"],
        generation_batch_size=cfg["generation_batch_size"],
        temperature=cfg["temperature"],
        beta=cfg["beta"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        num_train_epochs=cfg["epochs"],
        logging_steps=5,
        save_strategy="steps",
        save_steps=cfg["dev_eval_steps"],
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        seed=cfg["seed"],
        report_to="none",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg["gpu_memory_utilization"],
        vllm_max_model_length=cfg["vllm_max_model_length"],
    )
    callback = DevEvalCallback(reward, dev_rows, cfg)
    trainer = GRPOTrainer(model=model, args=args, train_dataset=dataset,
                          reward_funcs=reward, processing_class=tokenizer,
                          callbacks=[callback])
    callback.trainer = trainer
    trainer.train()
    trainer.model.save_pretrained(os.path.join(cfg["output_dir"], "lora_last"))
    if callback.history:
        print("done | best dev:", max(h["acc"] for h in callback.history))