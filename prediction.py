import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_lagged_features(df, lags=3):
    lagged_data = pd.DataFrame(index=df.index)
    for ticker in df.columns:
        for lag in range(1, lags+1):
            lagged_data[f'{ticker}_lag{ lag}'] = df[ticker].shift(lag)
    return lagged_data


#declare which stocks to track
tickers = {'^GSPC', '^IXIC', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META','ORCL','AMD','INTC',
           'VZ', 'TMUS', 'T'} #verizon, tmobile, A T&T

#set start and end dates for previous data
startDate = '2020-01-01'
endDate = '2025-07-05'

#pull data
raw_data = yf.download(tickers, start=startDate, end=endDate)
data = raw_data['Close']    
data.columns.name = None
print(data.head())
print(data.columns)
#drop rows with missing values
data.dropna(inplace=True)

#shift Apple's target data one day earlier so next day can be predicted based off rest
data['AAPL_Target'] = data['AAPL'].shift(-1)

#drop the last row since target is not a number
data = data.dropna()

#all current day prices
X = data.drop(columns=['AAPL_Target'])

#apple next day price
Y = data['AAPL_Target']

#split into training and testing data to train regression model
#X_test has the features, Y_test has is the true target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle =False, test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)

#use input to generate prediction for AAPL
y_pred = model.predict(X_test)
print("mean squared error: ", mean_squared_error(Y_test, y_pred))
print("R-squared: ", r2_score(Y_test, y_pred))

data_no_target= data.drop(columns=['AAPL_Target'])
lagged_features = create_lagged_features(data_no_target, lags=5)  

target = data['NVDA'].shift(-1)
full_data = lagged_features.copy()
full_data['target'] = target
full_data.dropna(inplace=True)

NVDA_X = full_data.drop(columns='target')
NVDA_Y = full_data['target']

NVDA_X_train, NVDA_X_test, NVDA_Y_train, NVDA_Y_test = train_test_split(NVDA_X, NVDA_Y, shuffle=False, test_size=0.2)

NVDA_model = LinearRegression()
NVDA_model.fit(NVDA_X_train, NVDA_Y_train)
NVDA_Y_pred = NVDA_model.predict(NVDA_X_test) 

print("NVDA mean squared error: ", mean_squared_error(NVDA_Y_test, NVDA_Y_pred))
print("NVDA R-squared: ", r2_score(NVDA_Y_test, NVDA_Y_pred))


AAPLfig, ax = plt.subplots()
#ax.figure(figsize=(10, 6))
ax.plot(Y_test.index, Y_test, label='Actual')
ax.plot(Y_test.index, y_pred, label='Predicted')
ax.set_title('AAPL: Actual vs Predicted Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True)
AAPLfig.tight_layout()
st.pyplot(AAPLfig)

NVDAfig, bx = plt.subplots()
#bx.figure(figsize=(10, 6))
bx.plot(NVDA_Y_test.index, NVDA_Y_test, label="Actual")
bx.plot(NVDA_Y_test.index, NVDA_Y_pred, label="Predicted")
bx.set_title("NVDA Price Prediction Using Lag Features")
bx.set_xlabel("Date")
bx.set_ylabel("Price")
bx.legend()
bx.grid(True)
NVDAfig.tight_layout()
st.pyplot(NVDAfig)