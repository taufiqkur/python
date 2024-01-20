import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

# Memuat dan Memproses Data
data = pd.read_csv('gt.csv')  # Gantilah 'nama_file.csv' dengan nama file dataset Anda
data['Date'] = pd.to_datetime(data['Date'])  # Mengkonversi format kolom 'Date' menjadi datetime

# Menambahkan MA
data['MA_20'] = data['Close'].rolling(window=20).mean()  # Rata-rata pergerakan harian
data['MA_10'] = data['Close'].rolling(window=10).mean()  # Rata-rata pergerakan mingguan
data['MA_5'] = data['Close'].rolling(window=5).mean()  # Rata-rata pergerakan bulanan

# Menambahkan RSI
def calculate_rsi(data, period=14):
    diff = data['Close'].diff(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

# Menambahkan Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    return upper_band, rolling_mean, lower_band

data['UpperBand'], data['MiddleBand'], data['LowerBand'] = calculate_bollinger_bands(data)

# Menambahkan MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()

    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    return macd, signal

data['MACD'], data['Signal_Line'] = calculate_macd(data)

# Ekstraksi fitur-fitur waktu yang relevan
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Membagi Data menjadi Data Latih dan Data Uji
X = data.drop(['Date', 'Close'], axis=1)  # Menghapus kolom 'Date' dan 'Target'
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun Model XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                         max_depth=5, alpha=10, n_estimators=10)
model.fit(X_train, y_train)

# Mengevaluasi Model
y_pred = model.predict(X_test)
# Menampilkan hasil prediksi untuk lima baris pertama dari data uji
print("Hasil Prediksi:")
print(pd.DataFrame({'Actual': y_test.head(), 'Predicted': y_pred[:5]}))

mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae}')

# Optimasi Model
param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f'Best Parameters: {best_params}')

# Gunakan model terbaik untuk prediksi
y_pred_optimized = best_model.predict(X_test)
# Menampilkan hasil prediksi teroptimalkan untuk lima baris pertama dari data uji
print("\nHasil Prediksi Teroptimalkan:")
print(pd.DataFrame({'Actual': y_test.head(), 'Predicted': y_pred_optimized[:5]}))

mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
print(f'Optimized MAE: {mae_optimized}')

# Memastikan panjang y_test dan y_pred_optimized sesuai
print("Panjang y_test:", len(y_test))
print("Panjang y_pred_optimized:", len(y_pred_optimized))

# Menampilkan Grafik Harga Saham Aktual dan Hasil Prediksi
plt.figure(figsize=(12, 6))
plt.plot(data['Date'].iloc[len(y_train):len(y_train)+len(y_test)], y_test, label='Harga Saham Aktual', marker='o')
plt.plot(data['Date'].iloc[len(y_train):], y_pred_optimized, label='Prediksi', marker='o')
plt.title('Grafik Harga Saham Aktual dan Prediksi')
plt.xlabel('Tanggal')
plt.ylabel('Harga Saham')
plt.legend()
plt.show()