# %%

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor

data = pd.read_csv(r'D:\data-cal-1.csv')
descriptors_list = ['barrier','nmr']
model_list = ['XGB']
reacts_list = data['reacts'].unique()


def testset(test_react):
    return data[data['reacts'] != test_react], data[data['reacts'] == test_react]

def descriptors_X(descriptors,data):
    if descriptors == 'barrier':
        return data[['T','Barrier']]
    elif descriptors == 'nmr':
        return data[['T','NH','N1','N2-1','N2-2','N2-3']]
    
def model(models):
    random_state = 42
    reg = ''
    if models == 'RF':
        reg = RandomForestRegressor(random_state=random_state, n_estimators=10)
    elif models == 'XGB':
        reg = XGBRegressor(random_state=random_state, n_estimators=1600, learning_rate=0.01)
    elif models == 'KNeighbors':
        reg = KNeighborsRegressor()
    elif models == 'AdaBoost':
        reg = AdaBoostRegressor(n_estimators=20, random_state=random_state)
    elif models == 'Linear':
        reg = LinearRegression()
    elif models == 'GradientBoost':
        reg = GradientBoostingRegressor( random_state=random_state, loss='squared_error',n_estimators=20)
    elif models == 'Lasso':
        reg = Lasso(alpha=0.01, random_state=random_state)
    elif models == 'ElasticNet':
        reg = ElasticNet(alpha=0.1, l1_ratio=0.1, random_state=random_state)
    elif models == 'DecisionTree':
        reg = DecisionTreeRegressor(random_state=random_state)
    elif models == 'ExtraTrees':
        reg = ExtraTreesRegressor(n_estimators=5,random_state=random_state)
    elif models == 'SVM':
        reg = SVR(kernel='rbf', C=1, gamma = 0.001)
    return reg

for models in model_list:
    reg = model(models)
    for descriptors in descriptors_list:
        r2_test_react_sum=0
        RMSE_sum = 0
        RMSE_all_react=[]
        for test_react in reacts_list:
            train_data,test_data = testset(test_react)

            X_train = descriptors_X(descriptors,train_data).values
            y_train = train_data['logK_cal'].values
            X_test = descriptors_X(descriptors,test_data).values
            y_test = test_data['logK_cal'].values

            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            r2_test_react = r2_score(y_pred,y_test)
            r2_test_react_sum +=  r2_test_react
            RMSE = mean_squared_error(y_pred, y_test, squared=False) 
            RMSE_sum += RMSE
            RMSE_all_react.append(RMSE)
 
            y_importances = reg.feature_importances_

        avg_RMSE = round(RMSE_sum / len(reacts_list),2)
        avg_r2_test_react = r2_test_react_sum / len(reacts_list)

        print(models,descriptors,avg_RMSE,avg_r2_test_react,RMSE_all_react,y_importances)
        
