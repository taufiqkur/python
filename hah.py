import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Assume you have a dataset 'stock_prices.csv' with the columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
# and you want to predict the 'Close' price for future dates

# Load the data
data = pd.read_csv('skrip.csv')

# Konversi kolom tanggal menjadi tipe data datetime
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Ekstrak fitur-fitur dari tanggal
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day

# Drop kolom tanggal asli dan kolom target
data = data.drop(['Date'], axis=1)
print(data.head())

# Handle nilai yang hilang jika ada
data = data.fillna(0)

# Define the feature matrix (X) and the target vector (y)
X = data.drop('Close', axis=1)
y = data['Close']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature matrix
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a xgboost regressor
regressor = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.05, max_depth=5, alpha=10, n_estimators=500)

# Train the model using the training set
regressor.fit(X_train, y_train)

# Make predictions on the testing set
predictions = regressor.predict(X_test)
print(predictions)

# Evaluate the model's performance using the root mean squared error
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print('Root Mean Squared Error: ', rmse)

# Simpan hasil prediksi ke dalam DataFrame
hasil_prediksi = pd.DataFrame({'Actual': y_test.values, 'Predicted': predictions})

# Simpan DataFrame ke dalam file Excel
#hasil_prediksi.to_excel('hasil_prediksi.xlsx', index=False)

# Visualisasi hasil prediksi
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted Harga Saham')
plt.xlabel('Data Point')
plt.ylabel('Harga Saham')
plt.legend()
plt.show()