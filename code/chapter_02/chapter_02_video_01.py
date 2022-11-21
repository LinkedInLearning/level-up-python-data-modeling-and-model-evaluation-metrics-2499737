import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

work_data = pd.read_csv(
  "/workspaces/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/level_up_data.csv"
  )

work_data.describe()

work_data.info()

imp_mean = IterativeImputer(random_state=0)

imputed_data = imp_mean.fit_transform(work_data.to_numpy())

pd.DataFrame(imputed_data).info()
