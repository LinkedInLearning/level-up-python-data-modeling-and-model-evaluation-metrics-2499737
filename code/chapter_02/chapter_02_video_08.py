from joblib import load
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree

X_train_scaled, X_test_scaled, y_train, y_test = load(
  '/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/model_data.joblib'
  )

param_distributions = {
  'max_features': range(1, 5), 
  'max_depth': range(5, 10)
  }

search = RandomizedSearchCV(
  estimator=tree.DecisionTreeClassifier(random_state=0),
  n_iter=5,
  param_distributions=param_distributions,
  random_state=0
  )

search.fit(X_scaled, y_train)

search.best_params_

search.score(X_test_scaled, y_test)

predictions = search.predict(X_test_scaled)

balanced_accuracy_score(y_test, predictions)
matthews_corrcoef(y_test, predictions)
roc_auc_score(y_test, search.predict_proba(X_test_scaled)[:, 1])

decision_tree = tree.DecisionTreeClassifier(max_features=3, max_depth=7)

decision_tree = decision_tree.fit(X_train_scaled, y_train)

plt.figure()

tree.plot_tree(decision_tree, filled=True)

plt.show()
