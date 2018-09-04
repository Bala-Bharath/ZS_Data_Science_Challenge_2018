import numpy as np
np.random.seed(34)
import pandas as pd

import math
from dateutil.parser import parse
import calendar
from datetime import datetime

test_file_path = '../dataset/yds_test2018.csv'

read_dir = '../dataset/'
write_dir = '../results/'
base_dir = '../'

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
scaler = None

exp_test, exp_pred = None, None
country, date = None, None

currency_code = {
        'Argentina':'ARS',
        'Belgium':'EUR',
        'Columbia':'COP',
        'Denmark':'DKK',
        'England':'GBP',
        'Finland':'EUR'
        } 

# history of monthly average currency rates taken for corresponding month
currency_rates = pd.read_excel('../currency_rate_history/currency_rates.xlsx')

def load_data():
    train = pd.read_csv(read_dir + 'yds_train2018.csv')
    test = pd.read_csv(test_file_path)
    expense = pd.read_csv(read_dir + 'promotional_expense.csv')
    expense.rename(columns={'Product_Type':'Product_ID'}, inplace=True)
    holiday = pd.read_excel(read_dir + 'holidays.xlsx')
    return train, test, expense, holiday

# converts all local currencies into usd with an appropriate monthly average currency rate
def convert_to_usd(data, columns):
    for col in columns:
        for i in data.index:
            if not math.isnan(data.loc[i, col]):
                dt = data.loc[i, 'Date'].isoformat()[:-3]
                code = currency_code[data.loc[i, 'Country']] 
                data.loc[i, col] *= currency_rates.loc[dt,code].values[0]
 
# converts back into local currencies from usd
def convert_from_usd(data, columns):
    for col in columns:
        for i in data.index:
            if not math.isnan(data.loc[i, col]):
                dt = data.loc[i, 'Date'].isoformat()[:-3]
                code = currency_code[data.loc[i, 'Country']] 
                data.loc[i, col] /= currency_rates.loc[dt,code].values[0]
                    
# splits a date (eg: 2018-02-03) into year, month, week number (eg: 2018, 1, 5)
def date_split(date_str):
    dt = parse(date_str)
    return dt.year, dt.month, dt.isocalendar()[1]

# attaches last day of month and converts into date (eg: 2018,01 --> 2018-01-31)
def get_date(val):
    day = calendar.monthrange(val[0],val[1])[1]
    return datetime.date(datetime(val[0], val[1], day))
    
def add_missing_columns(x, testx):    
    missing_cols = set(x.columns) - set(testx.columns)
    for col in missing_cols:
        testx[col] = 0
    testx = testx[x.columns]
    return testx

def SMAPE(f, a):
    return 200 * np.mean(abs(f-a) / (abs(a) + abs(f)))

def predict_expense_prices(data):
    global exp_test, exp_pred
    train = data[data['Expense_Price'].isnull() ^ True]
    test = data[data['Expense_Price'].isnull()]

    x = train.drop('Expense_Price', axis=1)
    y = train['Expense_Price']
    testx = test.drop('Expense_Price', axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    y_test.reset_index(drop=True, inplace=True)
    
    model = XGBRegressor(n_estimators=100,
                      max_depth=15,
                      colsample_bylevel=0.8,
                      colsample_bytree=1,
                      booster='dart'
                      )
    
    model.fit(x, y)    
    pred = model.predict(testx)
    data.loc[testx.index, 'Expense_Price'] = pred
    
# fill the expense_prices of test data with moving average filtered by country and month
def fill_expense_prices(data):
    for i in data.index:
        if math.isnan(data.loc[i,'Expense_Price']):
            prev_exp_prices = data[(data['Country'] == data.loc[i,'Country']) &
                                   (data['Month'] == data.loc[i,'Month'])
                                   ]['Expense_Price']
            data.loc[i,'Expense_Price'] = np.nanmean(prev_exp_prices)
            
def get_train_test_data(data, test_size=0.3):
    split_at = int(data.shape[0] * (1-test_size))
    train = data.iloc[:split_at, :]
    test = data.iloc[split_at:, :]
    
    x_train = train.drop('Sales', axis=1)
    y_train = train['Sales']
    x_test = test.drop('Sales', axis=1)
    y_test = test['Sales']
    
    return x_train, x_test, y_train, y_test

# preprocessing holiday data
#   1. count number of holidays in a month
#   2. specify a day is holiday or not (1 or 0)
#   3. group by year, month, and country.
def get_holiday_p(holiday):
    holiday_p = holiday.copy()
    t = holiday['Date'].apply(date_split)
    holiday_p['Year'] = [dt[0] for dt in t]
    holiday_p['Month'] = [dt[1] for dt in t]
    holiday_p['Week'] = [dt[2] for dt in t]

    for day in range(1, 32):
        holiday_p['Holiday_' + str(day)] = 0
    
    for i in range(holiday_p.shape[0]):
        day = parse(holiday_p.loc[i, 'Date']).day
        holiday_p.loc[i,'Holiday_' + str(day)] = 1
        
    holiday_p = holiday_p.groupby(by=['Year', 'Month','Country'], as_index=False).agg({
        'Holiday':len,
        'Holiday_1':sum,
        'Holiday_2':sum,
        'Holiday_3':sum,
        'Holiday_4':sum,
        'Holiday_5':sum,
        'Holiday_6':sum,
        'Holiday_7':sum,
        'Holiday_8':sum,
        'Holiday_9':sum,
        'Holiday_10':sum,
        'Holiday_11':sum,
        'Holiday_12':sum,
        'Holiday_13':sum,
        'Holiday_14':sum,
        'Holiday_15':sum,
        'Holiday_16':sum,
        'Holiday_17':sum,
        'Holiday_18':sum,
        'Holiday_19':sum,
        'Holiday_20':sum,
        'Holiday_21':sum,
        'Holiday_22':sum,
        'Holiday_23':sum,
        'Holiday_24':sum,
        'Holiday_25':sum,
        'Holiday_26':sum,
        'Holiday_27':sum,
        'Holiday_28':sum,
        'Holiday_29':sum,
        'Holiday_30':sum,
        'Holiday_31':sum
        })

    return holiday_p

def get_scaled_data(data, is_training = True):
    global scaler 
    if is_training:
        scaler = RobustScaler()
        return scaler.fit_transform(data)
    return scaler.transform(data)

train, test, expense, holiday = load_data()
holiday_p = get_holiday_p(holiday)

def get_train_data():
    train_data_p = train.groupby(by=['Year', 'Month', 'Product_ID','Country'], as_index=False).agg({
        'Sales':np.nansum
    })
    
    train_merge = pd.merge(train_data_p, holiday_p, on=['Year', 'Month','Country'], how='left')
    train_merge = pd.merge(train_merge, expense, on=['Year', 'Month', 'Product_ID','Country'], how='left')
    train_merge['Date'] = train_merge[['Year', 'Month']].apply(get_date, axis=1)
    
    train_merge['Holiday'].fillna(value = 0, inplace = True)
    for day in range(1,32):
        train_merge['Holiday_' + str(day)].fillna(value = 0, inplace = True)
        
    convert_to_usd(train_merge, columns=['Sales', 'Expense_Price'])
    
    global country, date
    country = train_merge['Country'].copy()
    date = train_merge['Date'].copy()
    
    train_merge = pd.get_dummies(train_merge, columns=['Country', 'Product_ID'], drop_first=True)
    train_merge.drop('Date', axis=1, inplace=True)

    predict_expense_prices(train_merge)
    return train_merge

def get_test_data():
    test_merge = pd.merge(test, holiday_p, on=['Year', 'Month','Country'], how='left')
    test_merge = pd.merge(test_merge, expense, on=['Year', 'Month', 'Product_ID','Country'], how='left')
    test_merge['Date'] = test_merge[['Year', 'Month']].apply(get_date, axis=1)
    
    test_merge['Holiday'].fillna(value = 0, inplace = True)
    for day in range(1,32):
        test_merge['Holiday_' + str(day)].fillna(value = 0, inplace = True)

    convert_to_usd(test_merge, columns=['Expense_Price'])
    fill_expense_prices(test_merge)
    testx = pd.get_dummies(test_merge, columns=['Country', 'Product_ID'], drop_first=True)
    testx.drop(columns=['S_No','Date','Sales'], axis=1, inplace=True)
    
    return testx, test_merge
