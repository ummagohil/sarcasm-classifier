# Sarcasm Classifier

A small learning project for **SemEval-2018 Task 3, Subtask A**: classifying
English tweets as ironic or non-ironic. It uses the Hugging Face
`cardiffnlp/tweet_eval` **irony** configuration, which republishes this task in
a convenient standard format, and a TF-IDF + logistic regression model.

> Sarcasm and irony overlap, but they are not identical. The dataset labels
> irony, so this project is a useful sarcasm baseline—not a perfect detector of
> every kind of sarcasm.

## What you will build

The project follows this pipeline:

1. Download labelled tweets from Hugging Face.
2. Turn words and two-word phrases into TF-IDF numeric features.
3. Train logistic regression on the training split.
4. evaluate only on the held-out test split.
5. Print precision, recall, F1, a confusion matrix, and sample predictions.
6. Save the model and use it on your own sentence.

The dataset is intentionally small. The original shared task supplied 3,834
training tweets and 784 test tweets. TweetEval divides the original training
material into training and validation splits while retaining the held-out test
split.

## 1. Python on this Mac

Your machine already has Python 3.13 at `/usr/local/bin/python3`. Confirm it:

```bash
python3 --version
```

If Python is missing on another Mac, install Homebrew from
<https://brew.sh>, then run:

```bash
brew install python
python3 --version
```

Do not install project packages globally. A virtual environment keeps this
project's packages isolated from the rest of the machine.

## 2. Open the project and set up the repository

```bash
cd ~/Documents/sarcasm-classifier
git init -b main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, `python` refers to the isolated project interpreter. Your
terminal prompt normally shows `(.venv)`. To leave it later, run `deactivate`.

Make the first commit after Git is initialized:

```bash
git add .
git commit -m "Set up baseline sarcasm classifier"
```

The `.gitignore` prevents the environment, downloaded data, and generated model
from being committed. These can all be recreated from the source code.

## 3. Import and inspect the dataset first

```bash
python download_data.py
```

`load_dataset("cardiffnlp/tweet_eval", "irony")` downloads the dataset through
the Hugging Face `datasets` library. The script caches a local copy under
`data/tweet_eval_irony`, prints the split sizes and label names, and displays
five examples. Labels are `0` for non-irony and `1` for irony.

## 4. Train and print the results

```bash
python train.py
```

TF-IDF gives more weight to informative words or word pairs and less weight to
terms that appear everywhere. Logistic regression then learns a weight for each
feature. The script trains only on `train` and measures performance on `test`,
which the model has not seen during training.

The printed metrics mean:

- **precision:** when the model predicts irony, how often it is right;
- **recall:** how much of the actual irony the model finds;
- **F1:** a balance of precision and recall;
- **accuracy:** the fraction of all predictions that are correct;
- **confusion matrix:** counts of correct and incorrect predictions by class.

The trained pipeline is saved to `models/sarcasm_pipeline.joblib`.
The baseline result from the first verified run is recorded in `RESULTS.md`.

## 5. Try your own text

```bash
python predict.py "Oh great, another meeting that could have been an email"
python predict.py "I enjoyed walking in the park today"
```

Probabilities are confidence-like scores, not guarantees. Sarcasm depends on
speaker intent and context, which may not be present in one tweet. The dataset
may also encode demographic, topic, and annotation biases.

## Project files

- `download_data.py` downloads and previews the data.
- `train.py` creates, evaluates, prints, and saves the classifier.
- `predict.py` runs the saved classifier on new text.
- `RESULTS.md` records the first held-out evaluation.
- `requirements.txt` records Python dependencies.
- `.gitignore` excludes generated files.

## Good next experiments

Try changing `ngram_range`, inspecting incorrect predictions, or comparing this
baseline with a small pretrained transformer. Keep the test split untouched
while tuning: use TweetEval's validation split to choose settings, then use the
test split once for the final report.

## Dataset

The task and dataset were introduced by Van Hee, Lefever, and Hoste (2018):
<https://aclanthology.org/S18-1005/>. TweetEval was introduced by Barbieri et
al. (2020). Hugging Face dataset card:
<https://huggingface.co/datasets/cardiffnlp/tweet_eval>
