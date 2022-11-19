from imblearn.combine import SMOTEENN
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

work_data = pd.read_csv(
  "/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/level_up_data.csv"
  )

work_data = work_data.sample(frac = .1)

encode_cats = pd.get_dummies(work_data['department'], prefix='department')

work_data = work_data.drop({'department'}, axis=1)

work_data = work_data.join(encode_cats)

predictors = work_data.drop('separatedNY', axis=1)

outcome = work_data['separatedNY']

imp_mean = IterativeImputer(random_state=1001)

imputed_data = imp_mean.fit_transform(predictors)

smote_enn = SMOTEENN(random_state=1001)

balanced_data, balanced_outcome = smote_enn.fit_resample(imputed_data, outcome)  

X_train, X_test, y_train, y_test = train_test_split(
    balanced_data, balanced_outcome, test_size=0.3, random_state=1001
    )

X_train_scaler = StandardScaler().fit(X_train)

X_train_scaled = X_train_scaler.transform(X_train)

X_test_scaler = StandardScaler().fit(X_test)

X_test_scaled = X_test_scaler.transform(X_test)

dump(
  [X_train_scaled, X_test_scaled, y_train, y_test], 
  '/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/model_data.joblib'
  )
