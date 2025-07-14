import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

def prepare_data(data, look_back=60):
    dataset = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
    
    return np.array(x_train), np.array(y_train), scaler, dataset

def predict_future(model, data, scaler, look_back=60, days=30):
    last_sequence = data[-look_back:]
    last_sequence_scaled = scaler.transform(last_sequence)
    
    predictions = []
    current_sequence = last_sequence_scaled.copy()
    
    for _ in range(days):
        next_pred = model.predict(current_sequence.reshape(1, look_back, 1))
        predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))