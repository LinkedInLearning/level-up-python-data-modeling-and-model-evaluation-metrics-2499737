import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc

X, y = load_breast_cancer(return_X_y=True)

clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)

roc_auc_score(y, clf.predict_proba(X)[:, 1])

roc_auc_score(y, clf.predict(X))
