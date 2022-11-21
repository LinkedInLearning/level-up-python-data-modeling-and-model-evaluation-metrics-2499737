from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import numpy as np
import pandas as pd

work_data = pd.read_csv(
  "/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/level_up_data.csv"
  )

encode_cats = pd.get_dummies(work_data[{'department'}])

work_data = work_data.drop({'department'}, axis=1)

work_data = work_data.join(encode_cats)

predictors = work_data.drop('separatedNY', axis=1)

outcome = work_data['separatedNY']

Counter(outcome)

smote_enn = SMOTEENN(random_state=1001)

X, y = smote_enn.fit_resample(predictors, outcome)  

Counter(y)

samplers = [
  SMOTE(random_state=0), 
  SMOTEENN(random_state=0), 
  SMOTETomek(random_state=0)
  ]

for i in range(samplers.__len__()):
  X, y = samplers[i].fit_resample(predictors, outcome)  
  print(Counter(y))
