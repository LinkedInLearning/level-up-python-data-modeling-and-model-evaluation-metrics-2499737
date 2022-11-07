import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

work_data = pd.read_csv("/Users/sethberry/Documents/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/level_up_data.csv")

encode_cats = pd.get_dummies(work_data[{'department', 'job_level'}])

work_data = work_data.drop({'department', 'job_level'}, axis = 1)

work_data = work_data.join(encode_cats)

predictors = work_data.drop('separated_ny', axis=1)

outcome = work_data['separated_ny']

X_train, X_test, y_train, y_test = train_test_split(predictors, outcome, test_size=0.33, random_state=42)

scaler = StandardScaler().fit(X_train)

X_scaled = scaler.transform(X_train)

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

  regression.fit(X_scaled, y_train)

  pred = regression.predict(X_test)

  auc = roc_auc_score(y_test, pred)
  
  bas = balanced_accuracy_score(y_test, pred)
  
  mcc = matthews_corrcoef(y_test, pred)
  
  result_data = pd.DataFrame({'model': [model_name], 
  'auc': [auc], 
  'bas': bas, 
  'mcc': mcc})
  
  result.append(result_data)

pd.concat(result)
