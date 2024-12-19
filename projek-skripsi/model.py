import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import datetime
import matplotlib.pyplot as plt

# Fungsi untuk menghitung Relative Strength Index (RSI)
def calculate_rsi(data, period=14):
    diff = data['Close'].diff(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_future_rsi(full_data, period=14):
    # Menghitung selisih harga Close harian
    diff = full_data['Close'].diff()

    # Memisahkan gain dan loss
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)

    # Menghitung rata-rata eksponensial bergerak untuk gain dan loss
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    # Menghitung RS dan RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Fungsi untuk menghitung Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    return upper_band, rolling_mean, lower_band

# Fungsi untuk menghitung Bollinger Bands di Masa Depan
def calculate_future_bollinger_bands(data, window=20, num_std_dev=2, num_days=1):
    data_len = len(data)
    future_upper_band = pd.Series(index=range(data_len, data_len + num_days), dtype='float64')
    future_middle_band = pd.Series(index=range(data_len, data_len + num_days), dtype='float64')
    future_lower_band = pd.Series(index=range(data_len, data_len + num_days), dtype='float64')

    if data_len == 0:
        # Jika data kosong, kembalikan Series kosong
        return future_upper_band, future_middle_band, future_lower_band

    # Gunakan nilai Close di masa depan untuk menghitung Bollinger Bands
    close_values = data['Close'].append(pd.Series(index=future_upper_band.index))

    # Ambil window + 1 nilai terakhir dari 'Close' untuk perhitungan Bollinger Bands
    close_values_for_calculation = close_values.iloc[-(window + 1):]

    # Hitung rolling mean dan rolling standard deviation
    rolling_mean = close_values_for_calculation.rolling(window=window).mean()
    rolling_std = close_values_for_calculation.rolling(window=window).std()

    # Hitung Upper Band, Middle Band, dan Lower Band
    future_upper_band_values = rolling_mean + (rolling_std * num_std_dev)
    future_middle_band_values = rolling_mean
    future_lower_band_values = rolling_mean - (rolling_std * num_std_dev)

    # Isi nilai Upper Band, Middle Band, dan Lower Band di masa depan
    future_upper_band = future_upper_band.fillna(future_upper_band_values.iloc[-1])
    future_middle_band = future_middle_band.fillna(future_middle_band_values.iloc[-1])
    future_lower_band = future_lower_band.fillna(future_lower_band_values.iloc[-1])

    return future_upper_band, future_middle_band, future_lower_band

# Fungsi untuk menghitung MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()

    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    return macd, signal

# Fungsi untuk menghitung MACD di Masa Depan
def calculate_future_macd(data, short_window=12, long_window=26, signal_window=9, num_days=1):
    data_len = len(data)
    future_macd = pd.Series(index=range(data_len, data_len + num_days))
    future_signal = pd.Series(index=range(data_len, data_len + num_days))

    if data_len == 0:
        # Jika data kosong, kembalikan Series kosong
        return future_macd, future_signal

    # Ambil nilai historis 'Close' untuk menghitung MACD di masa depan
    historical_close = data['Close']

    # Hitung EMA untuk short window
    short_ema = historical_close.ewm(span=short_window, adjust=False).mean()

    # Hitung EMA untuk long window
    long_ema = historical_close.ewm(span=long_window, adjust=False).mean()

    # Hitung MACD
    future_macd_values = short_ema - long_ema

    # Hitung EMA untuk sinyal
    future_signal_values = future_macd_values.ewm(span=signal_window, adjust=False).mean()

    # Isi nilai MACD dan sinyal di masa depan
    future_macd = future_macd_values.iloc[-num_days:]
    future_signal = future_signal_values.iloc[-num_days:]

    return future_macd, future_signal

# Fungsi untuk menghitung Moving Averages
def calculate_moving_averages(data, short_window=5, medium_window=10, long_window=20):
    data['MA_5'] = data['Close'].rolling(window=short_window).mean()
    data['MA_10'] = data['Close'].rolling(window=medium_window).mean()
    data['MA_20'] = data['Close'].rolling(window=long_window).mean()
    return data


def calculate_future_moving_averages(data, future_data,):
    full_data = pd.concat([data, future_data])
    
    # Menghitung moving averages dengan data lengkap
    full_data['MA_5'] = full_data['Close'].rolling(window=5).mean()
    full_data['MA_10'] = full_data['Close'].rolling(window=10).mean()
    full_data['MA_20'] = full_data['Close'].rolling(window=20).mean()
    
    # Hanya kembalikan bagian dari future_data dengan MA yang dihitung
    return full_data.iloc[len(data):].reset_index(drop=True)

# Fungsi untuk memprediksi masa depan dengan fitur-fitur teknikal
def predict_future_with_technical_features(best_model, input_data, start_date, end_date):
    # Membuat salinan data input untuk menghindari perubahan langsung
    data_copy = input_data.copy()

    # Membuat dataframe kosong untuk menyimpan hasil prediksi masa depan
    num_days = (end_date - pd.to_datetime(data_copy['Date'].max() + pd.DateOffset(1))).days
    future_dates = pd.date_range(data_copy['Date'].max() + pd.DateOffset(1), periods=num_days)

    # Inisialisasi dataframe masa depan dengan tanggal dan asumsi nilai close terakhir dari data historis
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Close': [data_copy['Close'].iloc[-1]] * len(future_dates),
        'Volume': [data_copy['Volume'].mean()] * len(future_dates)  # Menambahkan nilai rata-rata volume
    })

    # Menambahkan fitur-fitur waktu yang relevan pada data masa depan
    future_data['Year'] = future_data['Date'].dt.year
    future_data['Month'] = future_data['Date'].dt.month
    future_data['Day'] = future_data['Date'].dt.day
    future_data['DayOfWeek'] = future_data['Date'].dt.dayofweek

    # Menambahkan fitur-fitur teknikal ke future_data
    # Hitung RSI untuk data lengkap
    full_data = pd.concat([data_copy, future_data])
    future_data['RSI'] = calculate_future_rsi(full_data).iloc[len(data_copy):].reset_index(drop=True)

    # Menambahkan fitur-fitur MACD untuk masa depan
    future_macd_values, future_signal_values = calculate_future_macd(data_copy, short_window=12, long_window=26, signal_window=9, num_days=num_days)
    future_macd_values, future_signal_values = future_macd_values.tail(len(future_data)).values, future_signal_values.tail(len(future_data)).values

    # Sesuaikan panjang data MACD dan Signal Line untuk memastikan konsistensi
    future_data['MACD'] = future_macd_values
    future_data['Signal_Line'] = future_signal_values
    
    # Hitung MA untuk data masa depan menggunakan data historis dan data masa depan
    future_data_with_moving_averages = calculate_future_moving_averages(data_copy, future_data)
    
    # Update future_data dengan moving averages yang baru dihitung
    future_data['MA_5'] = future_data_with_moving_averages['MA_5']
    future_data['MA_10'] = future_data_with_moving_averages['MA_10']
    future_data['MA_20'] = future_data_with_moving_averages['MA_20']

    # Kolom yang relevan untuk model
    feature_columns = ['Volume', 'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'Signal_Line', 'Year', 'Month', 'Day', 'DayOfWeek']

    # Rolling prediction: menggunakan hasil prediksi sebagai input untuk prediksi berikutnya
    for i in range(1, len(future_data)):
        features_for_prediction = future_data.loc[i-1, feature_columns].values.reshape(1, -1)
        predicted_close = best_model.predict(features_for_prediction)
        future_data.loc[i, 'Close'] = predicted_close[0]

    # Menambahkan kolom hasil prediksi
    future_data['Close(Predicted)'] = future_data['Close']

    return future_data

# Fungsi untuk memuat model terbaik
def load_best_model(model_path):
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=10)
    model.load_model(model_path)
    return model