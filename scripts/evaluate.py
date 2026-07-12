import argparse
import json

from src.evaluation.evaluate import evaluate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preds = json.load(open(args.preds))
    _, report = evaluate_predictions(preds, args.data_root, args.out)

    print(f"overall {report['overall']['accuracy']:.4f}")
    for source, r in report["per_source"].items():
        print(f"{source} {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()