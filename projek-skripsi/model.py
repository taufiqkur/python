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

# Fungsi untuk menghitung Moving Averages di Masa Depan
def calculate_future_moving_averages(data, num_days=1):
    data_copy = data.copy()

    # Menambahkan MA_5 untuk num_days ke depan
    data_copy['MA_5'] = data_copy['Close'].rolling(window=5).mean().shift(-5 * num_days + 1)

    # Menambahkan MA_10 untuk num_days ke depan
    data_copy['MA_10'] = data_copy['Close'].rolling(window=10).mean().shift(-10 * num_days + 1)

    # Menambahkan MA_20 untuk num_days ke depan
    data_copy['MA_20'] = data_copy['Close'].rolling(window=20).mean().shift(-20 * num_days + 1)

    return data_copy

# Fungsi untuk memprediksi masa depan dengan fitur-fitur teknikal
# Fungsi untuk memprediksi masa depan dengan fitur-fitur teknikal
def predict_future_with_technical_features(best_model, input_data, start_date, end_date):
    # Menghitung jumlah hari antara end_date dan tanggal terakhir dalam data historis
    next_date = (pd.to_datetime(input_data['Date'].max()) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    num_days = (end_date - pd.to_datetime(input_data['Date'].max())).days

    # Membuat dataframe kosong untuk menyimpan hasil prediksi masa depan
    future_data = pd.DataFrame({'Date': pd.date_range(input_data['Date'].max() + pd.Timedelta(days=1), periods=num_days)})

    # Menambahkan fitur-fitur waktu yang relevan pada data masa depan
    future_data['Year'] = future_data['Date'].dt.year
    future_data['Month'] = future_data['Date'].dt.month
    future_data['Day'] = future_data['Date'].dt.day
    future_data['DayOfWeek'] = future_data['Date'].dt.dayofweek

    # Menambahkan fitur-fitur teknikal untuk masa depan
    rsi_values = calculate_rsi(pd.concat([input_data, future_data]), period=14).tail(len(future_data)).values

    # Menambahkan fitur-fitur MACD untuk masa depan
    future_macd_values, future_signal_values = calculate_future_macd(input_data, short_window=12, long_window=26, signal_window=9, num_days=num_days)
    future_macd_values, future_signal_values = future_macd_values.tail(len(future_data)).values, future_signal_values.tail(len(future_data)).values

    # Sesuaikan panjang data MACD dan Signal Line untuk memastikan konsistensi
    future_data['MACD'] = future_macd_values
    future_data['Signal_Line'] = future_signal_values

    # Sesuaikan panjang data untuk memastikan konsistensi
    rsi_values = rsi_values[-len(future_data):]

    # Menambahkan fitur-fitur teknikal ke future_data
    future_data['RSI'] = rsi_values

    # Menambahkan kolom 'Close' ke future_data
    future_data['Close'] = input_data['Close'].iloc[-1]  # Menggunakan nilai terakhir dari 'Close' pada data historis

    # Menambahkan fitur-fitur moving averages untuk masa depan
    future_data_with_moving_averages = calculate_future_moving_averages(input_data, num_days)
    future_data['MA_5'] = future_data_with_moving_averages['MA_5']
    future_data['MA_10'] = future_data_with_moving_averages['MA_10']
    future_data['MA_20'] = future_data_with_moving_averages['MA_20']

    # Menambahkan kolom 'Volume' dengan nilai rata-rata volume historis
    future_data['Volume'] = input_data['Volume'].mean()

    # Sesuaikan kolom agar sesuai dengan model yang ada
    # Pastikan urutan kolom sama dengan yang digunakan selama pelatihan
    future_data = future_data[['Date', 'Volume', 'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'Signal_Line', 'Year', 'Month', 'Day', 'DayOfWeek']]

    # Menggunakan model untuk memprediksi harga masa depan
    future_pred = best_model.predict(future_data.drop('Date', axis=1))

    # Menambahkan kolom hasil prediksi ke dalam dataframe
    future_data['Close(Predicted)'] = future_pred

    return future_data

# Fungsi untuk memuat model terbaik
def load_best_model(model_path):
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=10)
    model.load_model(model_path)
    return model