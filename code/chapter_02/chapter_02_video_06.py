import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

work_data = pd.read_csv("/Users/sethberry/Documents/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/level_up_data.csv")

encode_cats = pd.get_dummies(work_data[{'department', 'job_level'}])

work_data = work_data.drop({'department', 'job_level'}, axis = 1)

work_data = work_data.join(encode_cats)

predictors = work_data.drop('starting_salary', axis=1)

outcome = work_data['starting_salary']

X_train, X_test, y_train, y_test = train_test_split(predictors, outcome, test_size=0.33, random_state=42)

alpha_values = [0.1,0.3, 0.5, 0.8, 1]

model_list = dict([
  ('linear', linear_model.LinearRegression()), 
  ('ridge', linear_model.RidgeCV(alphas = alpha_values)), 
  ('lasso', linear_model.LassoCV(alphas = alpha_values)), 
  ('elastic', linear_model.ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1]))
  ])
  
result = []

for i in list(model_list.keys()):
  model_name = i
  
  regression = model_list.get(i)

  regression.fit(X_train, y_train)
  
  r2_score = regression.score(X_train, y_train)

  pred = regression.predict(X_test)

  mse = mean_squared_error(y_test, pred)
  
  mae = mean_absolute_error(y_test, pred)
  
  result_data = pd.DataFrame({'model': [model_name], 
  'r2': [r2_score],
  'mse': [mse], 
  'rmse': np.sqrt(mse), 
  'mae': mae})
  
  result.append(result_data)

pd.concat(result)
