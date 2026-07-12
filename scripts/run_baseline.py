import argparse
import json

from src.evaluation.generate import generate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=16000)
    args = parser.parse_args()

    rows = json.load(open(args.test_file))
    preds = generate_predictions(args.model, rows, args.out,
                                 max_tokens=args.max_tokens)
    print(f"saved {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()