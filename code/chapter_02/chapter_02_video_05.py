from joblib import load
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

X_train_scaled, X_test_scaled, y_train, y_test = load(
  '/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/model_data.joblib'
  )

svc_params = {
  'C': np.linspace(.1, 5, 3), 'kernel': ['linear', 'rbf']
  }

grid_search = GridSearchCV(
  estimator=svm.SVC(),
  param_grid=svc_params
  )
  
grid_search.fit(X_train_scaled, y_train)

pd.DataFrame(grid_search.cv_results_)

grid_predictions = grid_search.predict(X_test_scaled)

balanced_accuracy_score(y_test, grid_predictions)
matthews_corrcoef(y_test, grid_predictions)
roc_auc_score(y_test, grid_predictions)

random_search = RandomizedSearchCV(
  estimator=svm.SVC(),
  param_distributions=svc_params,
  n_iter=5,
  random_state=1001
  )

random_search.fit(X_train_scaled, y_train)

pd.DataFrame(random_search.cv_results_)

random_predictions = random_search.predict(X_test_scaled)

balanced_accuracy_score(y_test, random_predictions)
matthews_corrcoef(y_test, random_predictions)
roc_auc_score(y_test, random_predictions)
