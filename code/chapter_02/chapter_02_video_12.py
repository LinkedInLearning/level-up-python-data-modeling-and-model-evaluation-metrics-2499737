from joblib import load, dump
from imblearn.combine import SMOTEENN
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

link = 'https://2499737.youcanlearnit.net/tabulardata.html'

test_data = pd.read_html(link, match='department', header = 0)

test_data = test_data[0]

test_data = test_data.drop('Unnamed: 10', axis=1)

def department_encode(data_name):
  encode_cats = pd.get_dummies(data_name['department'], prefix='department')
  new_data = data_name.drop({'department'}, axis=1)
  encoded_data = new_data.join(encode_cats)
  return encoded_data

def predictor_outcome_split(data_name):
  predictors = data_name.drop('separatedNY', axis=1)
  outcome = data_name['separatedNY']
  return predictors, outcome

predictors, outcome = test_data.pipe(department_encode).pipe(predictor_outcome_split)

def impute_function(predictors_df):
  imp_mean = IterativeImputer(random_state=1001)
  imputed_data = imp_mean.fit_transform(predictors_df)
  return imputed_data

imputed_predictors = impute_function(predictors)

def smote_balance_function(imputed_data_name, outcome_name):
  smote_enn = SMOTEENN(random_state=1001)
  balanced_data, balanced_outcome = smote_enn.fit_resample(
    imputed_data_name, outcome_name
    )
  return balanced_data, balanced_outcome

balanced_data, balanced_outcome = smote_balance_function(imputed_predictors, outcome)

X_test_scaler = StandardScaler().fit(balanced_data)

X_test_scaled = X_test_scaler.transform(balanced_data)

xgb_model = load(
  '/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/xgboost_model.joblib'
  )
  
predictions = xgb_model.predict(X_test_scaled)

confusion_matrix(balanced_outcome, predictions)
