"""Use the saved classifier on your own text."""

import argparse
from pathlib import Path

import joblib


MODEL_PATH = Path("models/sarcasm_pipeline.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a sentence for sarcasm/irony")
    parser.add_argument("text", nargs="+", help="Text to classify")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run `python train.py` first.")

    text = " ".join(args.text)
    model = joblib.load(MODEL_PATH)
    probability = float(model.predict_proba([text])[0, 1])
    label = "sarcastic/ironic" if probability >= 0.5 else "not sarcastic/ironic"

    print(f"Text: {text}")
    print(f"Prediction: {label}")
    print(f"Sarcasm/irony probability: {probability:.3f}")


if __name__ == "__main__":
    main()

