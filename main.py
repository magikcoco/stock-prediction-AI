import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import yfinance as yf
import os
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from stocksymbol import StockSymbol


def download_ticker_data(ticker_list, start_date=None, end_date=None):
    all_data = {}
    for ticker in ticker_list:
        try:
            data = yf.download(ticker, start_date=start_date, end=end_date)
            if not data.empty:
                all_data[ticker] = data
            time.sleep(10)
        except Exception as e:
            print(f"Error downloading data for {ticker}. Error: {str(e)}")
            time.sleep(60)
    return all_data


def save_data_to_csv(all_data, output_folder="."):
    for ticker, data in all_data.items():
        filename = os.path.join(output_folder, f"{ticker}.csv")
        data.to_csv(filename)


def get_ticker_list(save_to_file=True, load_from_file=True):
    filename = "ticker_list.txt"
    no_file = not load_from_file
    companies = []
    if load_from_file:
        if os.path.exists("ticker_list.txt"):
            with open(filename, "r") as file:
                companies = [line.strip() for line in file]
        else:
            no_file = True
    if no_file:
        with open("api_key.txt", "r") as file:
            api_key = file.read().strip()
        ss = StockSymbol(api_key)
        symbol_list_us = ss.get_symbol_list(market="US")  # "us" or "america" will also work
        for company in symbol_list_us:
            if company.get('symbol') is not None:
                companies.append(company.get('symbol'))
    if save_to_file:
        with open("ticker_list.txt", "w") as file:
            for company in companies:
                file.write(f"{company}\n")
    return companies

'''
# Load Data
company = ['NVDA']  # TODO: do every company
start = dt.datetime(2012, 1, 1)  # Timestamp start, 1st of january
end = dt.datetime(2020, 1, 1)  # Timestamp end, 1st of january
data = yf.download(company, start=start, end=end)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))  # scale down all values, so they fit between 0 and 1
# this will only predict the closing price
# TODO: predict open prices, high prices, etc
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
prediction_days = 60  # how many days to base prediction on
x_train = []
y_train = []
for x in range(prediction_days, len(scaled_data)):
    # we need 60 values, and then a 61st "correct answer", so the AI can learn to predict
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # predicts next closing value here

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)
model.save(os.path.join('models', company[0]))
#model.load()

# Evaluate the Model Accuracy
# Load test data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()
test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make Predictions
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

# Plot test predictions
plt.plot(actual_prices, color='black', label=f"Actual {company} Price")
plt.plot(prediction_prices, color='green', label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()
'''
