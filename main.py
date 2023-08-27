import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Load Data
company = ['GOOG']  # TODO: do every company
start = dt.datetime(2012, 1, 1)  # Timestamp start, 1st of january
end = dt.datetime(2020, 1, 1)  # Timestamp end, 1st of january
#source = 'yahoo'
#yf.pdr_override()
data = yf.download(company, start=start, end=end)
#data = web.get_data_yahoo(company, start=start, end=end).reset_index()
#data = web.DataReader(company, data_source=source, start=start, end=end).reset_index()

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
#model.save()
#model.load()

# Evaluate the Model Accuracy
# Load test data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()
test_data = yf.download(company, start=test_start, end=test_end)
#test_data = web.DataReader(company, 'yahoo', test_start, test_end)
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
