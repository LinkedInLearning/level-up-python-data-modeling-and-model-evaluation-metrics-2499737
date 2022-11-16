from collections import Counter
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

work_data = pd.read_csv("/Users/sethberry/Documents/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/level_up_data.csv")

encode_cats = pd.get_dummies(work_data[{'department'}])

work_data = work_data.drop({'department'}, axis=1)

work_data = work_data.join(encode_cats)

predictors = work_data.drop('separatedNY', axis=1)

outcome = work_data['separatedNY']

Counter(outcome)

oversample = SMOTE()

X, y = oversample.fit_resample(predictors, outcome)  

Counter(y)
