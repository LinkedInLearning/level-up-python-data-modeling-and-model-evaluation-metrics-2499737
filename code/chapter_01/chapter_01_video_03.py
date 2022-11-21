import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc

work_data = pd.read_csv(
  "/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/level_up_data.csv"
  )

y = work_data['separatedNY']

X = work_data['workDistance']

X = X.to_numpy()

X = X.reshape(-1, 1)

simple_logistic = LogisticRegression(solver="liblinear", random_state=1001)

simple_logistic.fit(X, y)

predicted_probs = simple_logistic.predict_proba(X)[:, 1]

predicted_class = simple_logistic.predict(X)

roc_auc_score(y, predicted_probs)

roc_auc_score(y, predicted_class)
