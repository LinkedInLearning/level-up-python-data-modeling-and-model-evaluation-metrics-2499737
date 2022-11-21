from joblib import load
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef

X_train_scaled, X_test_scaled, y_train, y_test = load(
  '/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/model_data.joblib'
  )

model_list = dict([
  ('logistic', linear_model.LogisticRegression(penalty='none')), 
  ('ridge', linear_model.LogisticRegressionCV(penalty='l2', solver='lbfgs')), 
  ('lasso', linear_model.LogisticRegressionCV(penalty='l1', solver='liblinear')), 
  ('elastic', linear_model.LogisticRegressionCV(penalty='elasticnet', l1_ratios = [.1, .5, .7, .9, .95, .99, 1], solver='saga'))
])
  
result = []

for i in list(model_list.keys()):
  model_name = i
  
  regression = model_list.get(i)

  regression.fit(X_train_scaled, y_train)
  
  print(regression.coef_)

  pred = regression.predict(X_test_scaled)
  
  accuracy = accuracy_score(y_test, pred)

  auc = roc_auc_score(y_test, pred)
  
  bas = balanced_accuracy_score(y_test, pred)
  
  mcc = matthews_corrcoef(y_test, pred)
  
  result_data = pd.DataFrame({'model': [model_name], 
  'accuracy': [accuracy],
  'auc': [auc], 
  'bas': bas, 
  'mcc': mcc})
  
  result.append(result_data)

pd.concat(result)
