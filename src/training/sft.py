from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def load_model(cfg):
    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg["base_model"], max_seq_length=cfg["max_length"],
        dtype=None, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0,
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth", random_state=cfg["seed"])
    return model, tokenizer


def train(cfg):
    model, tokenizer = load_model(cfg)
    data = load_dataset("json", data_files={
        "train": cfg["train_file"], "val": cfg["val_file"]})

    args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=cfg["warmup_ratio"],
        max_length=cfg["max_length"],
        packing=False,
        completion_only_loss=True,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        per_device_eval_batch_size=cfg["batch_size"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=cfg["eval_steps"],
        save_total_limit=2,
        bf16=True,
        seed=cfg["seed"],
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args,
                         train_dataset=data["train"], eval_dataset=data["val"],
                         processing_class=tokenizer)
    trainer.train()

    model.save_pretrained(cfg["adapter_dir"])
    tokenizer.save_pretrained(cfg["adapter_dir"])
    model.save_pretrained_merged(cfg["merged_dir"], tokenizer,
                                 save_method="merged_16bit")