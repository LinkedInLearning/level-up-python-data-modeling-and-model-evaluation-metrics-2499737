import pandas as pd
import numpy as np

regression_data = pd.read_csv('/Users/sethberry/Documents/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/data/regression_output.csv')

regression_data['absolute_difference'] = abs(regression_data['actual'] - 
  regression_data['predicted'])
  
regression_data['squared_error'] = (regression_data['actual'] - 
  regression_data['predicted'])**2   

mae = regression_data['absolute_difference'].mean()

rmse = np.sqrt(regression_data['squared_error'].mean())

mae

rmse

row_count = regression_data.__len__()
mae_sum = 0
rmse_sum = 0
  
for i in range(row_count):
    mae_sum += abs(regression_data['actual'][i] - regression_data['predicted'][i])
    rmse_sum += (regression_data['actual'][i] - regression_data['predicted'][i])**2
  
mae = mae_sum / row_count
rmse = np.sqrt(rmse_sum / row_count)

