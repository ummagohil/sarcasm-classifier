
from pathlib import Path

import joblib
from datasets import load_dataset, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/tweet_eval_irony")
MODEL_PATH = Path("models/sarcasm_pipeline.joblib")


def get_dataset():
    if DATA_PATH.exists():
        return load_from_disk(DATA_PATH)
    print("Local data not found; downloading cardiffnlp/tweet_eval (irony)...")
    return load_dataset("cardiffnlp/tweet_eval", "irony")


def main() -> None:
    dataset = get_dataset()

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, class_weight="balanced"),
            ),
        ]
    )

    print(f"Training on {len(dataset['train']):,} tweets...")
    model.fit(dataset["train"]["text"], dataset["train"]["label"])

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]
    predictions = model.predict(test_texts)

    print(f"\nEvaluating on {len(test_texts):,} held-out tweets")
    print("SemEval-2018 Task 3A labels: 0 = non-irony, 1 = irony")
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            target_names=["non-irony", "irony"],
            digits=3,
        )
    )
    print("Confusion matrix [[true 0], [true 1]]:")
    print(confusion_matrix(test_labels, predictions))

    print("\nA few predictions:")
    probabilities = model.predict_proba(test_texts)[:, 1]
    for text, actual, predicted, probability in list(
        zip(test_texts, test_labels, predictions, probabilities)
    )[:10]:
        print(
            f"actual={actual} predicted={predicted} "
            f"irony_probability={probability:.3f} | {text}"
        )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
