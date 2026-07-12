import argparse

import yaml

from src.training import sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sft.train(cfg)


if __name__ == "__main__":
    main()