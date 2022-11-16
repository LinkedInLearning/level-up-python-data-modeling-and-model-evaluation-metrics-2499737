from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

work_data = pd.read_csv("/Users/sethberry/Documents/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/level_up_data.csv")

work_data = work_data.sample(frac = .20)

encode_cats = pd.get_dummies(work_data[{'department'}])

work_data = work_data.drop({'department'}, axis=1)

work_data = work_data.join(encode_cats)

predictors = work_data.drop('separatedNY', axis=1)

outcome = work_data['separatedNY']

Counter(outcome)

imp_mean = IterativeImputer(random_state=1001)

imputed_data = imp_mean.fit_transform(predictors)

oversample = SMOTE()

X, y = oversample.fit_resample(imputed_data, outcome)  

Counter(y)

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.33, random_state=1001
  )

Counter(y_train)
Counter(y_test)

X_train_scaler = StandardScaler().fit(X_train)

X_train_scaled = X_train_scaler.transform(X_train)

X_test_scaler = StandardScaler().fit(X_test)

X_test_scaled = X_test_scaler.transform(X_test)

param_distributions = {'max_features': range(8, 10), 'ccp_alpha': range(0, 1)}

search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=1001),
                             n_iter=1,
                             param_distributions=param_distributions,
                             random_state=1001)
                             
search.fit(X_train_scaled, y_train)

search.best_params_

search.score(X_test_scaled, y_test)

cross_val_score()

predictions = search.predict(X_test_scaled)

Counter(predictions)

balanced_accuracy_score(y_test, predictions)
matthews_corrcoef(y_test, predictions)
roc_auc_score(y_test, predictions)

forest = RandomForestClassifier(max_features=8, ccp_alpha=0, random_state=0)

forest.fit(X_train_scaled, y_train)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

forest_importances = pd.Series(importances, index=predictors.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
