import numpy as np
np.random.seed(34)

import util

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from sklearn.preprocessing import RobustScaler

scaler = None

def get_scaled_data(data, is_training = True):
    global scaler 
    if is_training:
        scaler = RobustScaler()
        return scaler.fit_transform(data)
    return scaler.transform(data)

def print_scores(y_test, pred):
    x_test['Sales_true'] = y_test
    x_test['Sales_pred'] = pred
    x_test['Country'] = util.country[x_test.index].copy()
    x_test['Date'] = util.date[x_test.index].copy()
    util.convert_from_usd(x_test, columns=['Sales_true', 'Sales_pred'])
    y_test = x_test['Sales_true']
    pred = x_test['Sales_pred']

    x_test.drop(['Sales_true', 'Sales_pred', 'Country', 'Date'], axis=1, inplace=True)
    print(r2_score(y_test, pred))
    print(util.SMAPE(y_test, pred))
    
train_data = util.get_train_data()
x = train_data.drop('Sales', axis=1)
y = train_data['Sales']
testx, test_merge = util.get_test_data()

x_train, x_test, y_train, y_test = util.get_train_test_data(train_data, test_size=0.20)

model1 = RandomForestRegressor(n_estimators=500,
                               max_depth=25)
#model1.fit(x_train, y_train)
model1.fit(x, y)

model2 = GradientBoostingRegressor(n_estimators=300, 
                                  max_depth = 15,
                                  max_features = 0.9,
                                  min_impurity_decrease = 0.5)
#model2.fit(x_train, y_train)
model2.fit(x, y)

model3 = XGBRegressor(n_estimators=100,
                      max_depth=25,
                      colsample_bylevel=0.8,
                      colsample_bytree=1,
                      booster='dart',
                      )
model3.fit(x, y)

pred1 =  model1.predict(testx)
pred2 =  model2.predict(testx)
pred3 =  model3.predict(testx)
pred = pred1 + pred2  + pred3
pred /= 3

test_merge['Sales'] = pred
util.convert_from_usd(test_merge, columns=['Sales'])
test_merge[['S_No', 'Year', 'Month', 'Product_ID', 'Country', 'Sales']].to_csv(util.write_dir + 'predictions.csv', 
          index=False)