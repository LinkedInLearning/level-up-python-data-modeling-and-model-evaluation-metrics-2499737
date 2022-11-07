import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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

param_distributions = {'n_estimators': range(1, 5), 'max_depth': range(5, 10)}

search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=0),
                             n_iter=5,
                             param_distributions=param_distributions,
                             random_state=0)
                             
search.fit(X_train, y_train)

search.best_params_

search.score(X_test, y_test)

predictions = search.predict(X_test)

balanced_accuracy_score(y_test, predictions)
matthews_corrcoef(y_test, predictions)
roc_auc_score(y_test, predictions)
