import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

work_data = pd.read_csv(
  "/Users/sethberry/Documents/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/level_up_data.csv"
  )

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.30, random_state=1001
  )
