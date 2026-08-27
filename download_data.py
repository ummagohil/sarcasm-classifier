"""Download SemEval-2018 Task 3A via TweetEval and show its structure."""

from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "irony"
DATA_DIR = Path("data")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    dataset.save_to_disk(DATA_DIR / "tweet_eval_irony")

    print(dataset)
    print("\nLabel names:", dataset["train"].features["label"].names)
    print("\nFirst five training examples:")
    for row in dataset["train"].select(range(5)):
        print(f"  label={row['label']}  text={row['text']}")


if __name__ == "__main__":
    main()
