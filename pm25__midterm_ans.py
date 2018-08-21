import pandas as pd
import numpy as np
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, ensemble, preprocessing, metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
import xgboost as xgb
import datetime

datadir='/home/..../pm25/'

df = pd.read_csv(datadir+'df_0.0.7.csv')

X = df[['PM10', 'PM1', 'Temperature', 'Humidity',  
        'del_pm10', 'del_pm1', 'del_tem', 'del_hum',
        'lat', 'lon', 'hour']]

y = df[['PM2.5']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, random_state=40)

xgb1 = XGBRegressor()
xgb1 = XGBRegressor()
parameters = {#'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.5], #so called `eta` value
              'max_depth': [6],
              #'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [2000]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 4,
                        n_jobs = 1,
                        verbose=100)

xgb_grid.fit(X_train,
         y_train)


testdata=pd.read_csv(datadir+'test_1.0.0.csv')
test_x = testdata[['PM10', 'PM1', 'Temperature', 'Humidity',  
        'del_pm10', 'del_pm1', 'del_tem', 'del_hum',
        'lat', 'lon', 'hour']] 

test_pred = xgb_grid.predict(test_x)
test_ans = pd.DataFrame(test_pred)
avg_ans_df = test_ans.groupby(testdata['device_id'], as_index=True).mean()
testdatasheet=pd.read_csv( datadir+"/pm25/submission.csv")


testdatasheet['pred_pm25'] = np.array(avg_ans_df)
testdatasheet.to_csv('score/2.0.0.07.csv', index=False)