# Text-to-SQL with SFT and GRPO

Fine-tuning Qwen2.5-Coder-7B for text to SQL on BIRD and Spider.
The model is first trained with SFT, then with GRPO using a reward
that comes from running the generated SQL against the database.

## Results

Execution accuracy on a held out test set of 1600 examples from 21
databases that never appear in training.

| Model    | Overall | BIRD | Spider |
|----------|---------|------|--------|
| Base     | 50.6    | 38.1 | 67.1   |
| SFT      | 66.6    | 53.1 | 84.3   |
| SFT+GRPO | 68.4    | 55.1 | 85.9   |

## Data

Data comes from the OmniSQL formatted BIRD and Spider training sets.
Gold queries are run against their databases and rows are dropped when
the query errors, takes more than 15 seconds, or returns an empty
result. Splits are made at the database level so no test database
appears in training, stratified by source, 85/5/10 by row count.

## SFT

SFT trains a rank 16 LoRA on all projection modules with loss on the
completion only, for 2 epochs at learning rate 1e-4. The adapter is
saved on its own and also merged into the base model.

## GRPO

GRPO loads the merged SFT model and applies the saved SFT adapter on
top as the trainable adapter, so the SFT update is present twice at
the start. This matches the run that produced the numbers above.

Training uses TRL with vLLM in colocate mode. Each prompt gets 8
sampled completions at temperature 1.0. A completion is rewarded 1.0
when its result matches the gold result, 0.1 when the SQL runs but
gives a wrong result or times out, and 0 when it fails to run. Row
order is ignored unless the gold query has an ORDER BY.

The training pool keeps only prompts where the SFT model was neither
always right nor always wrong across 8 samples at temperature 1.0.
Training runs for 1 epochs. A fixed set of 150 dev examples is scored
every 50 steps and training stops after two checks without
improvement. The best and last adapters are saved. To resume a run,
point init_adapter at outputs/grpo/lora_best and set step_offset in
the config.

Query execution uses a sqlite progress handler as a wall clock guard,
which stops runaway queries mid statement where signal based timeouts
cannot.

## Setup

    pip install -r requirements.txt

Download the OmniSQL datasets and place them under data/omnisql.

## Usage

    python scripts/prepare_data.py --data-root data/omnisql
    python scripts/run_baseline.py --model Qwen/Qwen2.5-Coder-7B \
        --test-file outputs/data/test.json --out outputs/preds/base.json
    python scripts/evaluate.py --preds outputs/preds/base.json \
        --data-root data/omnisql --out outputs/results/base.json
    python scripts/train_sft.py --config configs/sft.yaml
    python scripts/build_grpo_pool.py --stage generate --model outputs/sft/merged \
        --train-file outputs/data/train.json --data-root data/omnisql
    python scripts/build_grpo_pool.py --stage bucket --model outputs/sft/merged \
        --train-file outputs/data/train.json --data-root data/omnisql
    python scripts/train_grpo.py --config configs/grpo.yaml