from joblib import load
import random
from sklearn.pipeline import Pipelineimport pandas as pd
import shap

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

xgb_model = load(
  '/Users/sethberry/Documents/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/xgboost_model.joblib'
  )
  
xgb_model.predict(balanced_outcome)
