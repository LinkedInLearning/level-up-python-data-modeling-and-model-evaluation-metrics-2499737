from joblib import load, dump
import numpy as np
import shap
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

X_train_scaled, X_test_scaled, y_train, y_test = load(
  '/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/model_data.joblib'
  )

xgb_model = xgb.XGBClassifier()

param_distributions = {
  'eta': np.linspace(0, 1, 11), 
  'gamma': range(0, 10, 2), 
  'min_child_weight': range(0, 10),
  'max_depth': range(3, 10), 
  'n_estimators': [50, 100, 200]
  }

xgb_search = RandomizedSearchCV(
  xgb_model, 
  param_distributions=param_distributions, 
  n_iter=5, 
  scoring='roc_auc', 
  random_state=1001
)

xgb_search.fit(X_train_scaled, y_train)

xgb_search.best_score_
xgb_search.best_params_
                             
xgb_search.score(X_test_scaled, y_test)

predictions = xgb_search.predict(X_test_scaled)

dump(
  xgb_search, 
  '/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/xgboost_model.joblib'
  )

balanced_accuracy_score(y_test, predictions)
matthews_corrcoef(y_test, predictions)
roc_auc_score(y_test, predictions)

Xd = xgb.DMatrix(X_train_scaled, label=y_train)

model = xgb.train(
  {'max_depth':6, 'n_estimators':200}, 
  Xd)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xd)

column_names = [
  'age', 'numberPriorJobs', 'proportion401K', 
  'startingSalary', 'currentSalary', 'performance', 
  'monthsToSeparate', 'workDistance', 'department_1', 
  'department_2', 'department_3'
  ]

shap.summary_plot(
  shap_values, 
  X_train_scaled,
  feature_names = column_names
  )


