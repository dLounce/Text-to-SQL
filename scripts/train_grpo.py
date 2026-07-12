import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.training import grpo
    grpo.train(cfg)


if __name__ == "__main__":
    main()