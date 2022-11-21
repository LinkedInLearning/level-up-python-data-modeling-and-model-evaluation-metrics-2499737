from joblib import load
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

X_train_scaled, X_test_scaled, y_train, y_test = load(
  '/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/model_data.joblib'
  )

param_distributions = {
  'max_features': range(5, 10), 
  'ccp_alpha': range(0, 3)
  }

search = RandomizedSearchCV(
  estimator=RandomForestClassifier(random_state=1001),
  n_iter=3,
  param_distributions=param_distributions,
  random_state=1001
  )
                             
search.fit(X_train_scaled, y_train)

search.best_params_

search.score(X_test_scaled, y_test)

predictions = search.predict(X_test_scaled)

Counter(predictions)

balanced_accuracy_score(y_test, predictions)
matthews_corrcoef(y_test, predictions)
roc_auc_score(y_test, search.predict_proba(X_test_scaled)[:, 1])

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
plt.show()
