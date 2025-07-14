from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import datetime
import pandas as pd

app = Flask(__name__)

def get_stock_data(ticker, start_date, end_date):
    '''creating flag to check for errors'''
    i = 0
    try:
        if not isinstance(ticker, str) or not ticker.isalpha():
            raise ValueError("Ticker must be a valid alphabetic symbol")
        i = 1
        start = pd.to_datetime(start_date)
        i = 2
        end = pd.to_datetime(end_date)
        i = 3

        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        i = 4

        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("No data returned for this ticker")
        i = 5

        # Handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            print("MultiIndex detected. Flattening columns...")
            data.columns = ['_'.join(col).strip() for col in data.columns.swaplevel()]
            i = 6
            # Try to find the Close column
            close_col = [col for col in data.columns if col.endswith('_Close') or col.startswith(f'{ticker}_Close')]
            if not close_col:
                raise ValueError(f"Flattened columns: {data.columns}. No '{ticker}_Close' column found")
            data['Close'] = data[close_col[0]]
        else:
            i = 7
            if 'Close' not in data.columns:
                raise ValueError(f"'Close' column not found. Columns: {list(data.columns)}")

        i = 8
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        if data['Close'].isnull().any():
            raise ValueError("Non-numeric values found in price data")
        i = 9

        return data[['Close']]
    
    except Exception as e:
        raise ValueError(f"Data download failed: {str(e)} (flag: {i})")

def prepare_data(data, look_back=60):
    
    try:
        
        if not isinstance(data, pd.DataFrame) or data.shape[1] != 1:
            raise ValueError("Input must be single-column DataFrame")
            
        prices = data.values.astype('float32')
        
        if prices.ndim != 2 or prices.shape[1] != 1:
            raise ValueError("Invalid array dimensions")
            
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)
        
        x_train, y_train = [], []
        for i in range(look_back, len(scaled_data)):
            seq = scaled_data[i-look_back:i, 0]
            if len(seq) == look_back:  
                x_train.append(seq)
                y_train.append(scaled_data[i, 0])
        
        x_train = np.array(x_train, dtype='float32').reshape(-1, look_back, 1)
        y_train = np.array(y_train, dtype='float32')
        
        return x_train, y_train, scaler, prices
    
    except Exception as e:
        raise ValueError(f"Data preparation failed: {str(e)}")

def predict_future(model, data, scaler, look_back=60, days=30):
    
    try:
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Invalid input data format")
            
        if data.shape[0] < look_back:
            raise ValueError("Insufficient historical data")
            
        last_sequence = data[-look_back:].reshape(-1, 1).astype('float32')
        last_sequence_scaled = scaler.transform(last_sequence)
        
        # Generate predictions
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(days):
            next_pred = model.predict(
                current_sequence.reshape(1, look_back, 1),
                verbose=0
            )[0, 0]
            
            predictions.append(float(next_pred))
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        
        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        return predictions.flatten()
    
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip().upper()
        if not ticker or not ticker.isalpha():
            return render_template('index.html', error="Invalid ticker symbol")
        
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365*5)  # 5 years data
        
        try:
            data = get_stock_data(ticker, start_date, end_date)
            
            look_back = 60
            x_train, y_train, scaler, dataset = prepare_data(data, look_back)
            
            # Build and train LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
            
            
            predictions = predict_future(model, dataset, scaler)
            
            historical_prices = {
                date.strftime('%Y-%m-%d'): float(price[0]) 
                for date, price in zip(data.index, data.values)
            }
            
            return render_template('index.html',
                                ticker=ticker,
                                historical=historical_prices,
                                predictions=predictions.tolist())
            
        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)