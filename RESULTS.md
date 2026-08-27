# Baseline Results

These results were produced locally with `python train.py` on 27 August 2026.
The model trained on 2,862 TweetEval/SemEval-2018 Task 3A tweets and was
evaluated on the official 784-tweet held-out test split.

## Classification report

```text
              precision    recall  f1-score   support

   non-irony      0.732     0.681     0.705       473
       irony      0.561     0.621     0.589       311

    accuracy                          0.657       784
   macro avg      0.646     0.651     0.647       784
weighted avg      0.664     0.657     0.659       784
```

## Confusion matrix

```text
[[322 151]
 [118 193]]
```

Rows are actual labels and columns are predictions. The model correctly found
193 ironic tweets and correctly rejected 322 non-ironic tweets. It missed 118
ironic tweets and incorrectly marked 151 non-ironic tweets as ironic.

This is a learning baseline rather than a state-of-the-art system. The official
SemEval paper reports a best Task A F1 of 0.71, though direct comparisons should
use the competition's exact metric and experimental protocol.
