# 🙊 Sarcasm Classifier

A small learning project for **SemEval-2018 Task 3, Subtask A**, classifying tweets as ironic or non-ironic. It uses the Hugging Face
`cardiffnlp/tweet_eval` **irony** configuration, which republishes this task in a convenient standard format and a TF-IDF + logistic regression model.

TF-IDF (Term Frequency–Inverse Document Frequency) converts important words into numerical features, and logistic regression uses those features to predict whether a tweet is ironic.

## 📑 The Set Up

The project follows this pipeline:

- Download labelled tweets from Hugging Face
- Turn words and two-word phrases into TF-IDF numeric features
- Train logistic regression on the training split
- Evaluate only on the held-out test split.
- Print precision, recall, F1, a confusion matrix, and sample predictions
- Save the model and use it on your own sentence

The dataset is intentionally small. The original shared task supplied 3,834
training tweets and 784 test tweets. TweetEval divides the original training
material into training and validation splits while retaining the held-out test
split.

## 💿 How to run the classifier

```bash
cd /sarcasm-classifier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_data.py
python train.py
python predict.py "Your example tweet"
```

## 🖥️ Project files

- `download_data.py` downloads and previews the data
- `train.py` creates, evaluates, prints, and saves the classifier
- `predict.py` runs the saved classifier on new text
- `RESULTS.md` records the first held-out evaluation
- `requirements.txt` records Python dependencies
- `.gitignore` excludes generated files.

## 📊 Dataset

- The task and dataset were introduced by Van Hee, Lefever, and Hoste (2018)
  <https://aclanthology.org/S18-1005/>

- TweetEval was introduced by Barbieri et
  al. (2020)
  <https://huggingface.co/datasets/cardiffnlp/tweet_eval>
