from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go  # Import Plotly library
import json
from model import load_best_model, predict_future_with_technical_features, calculate_moving_averages, calculate_rsi, calculate_macd

# Inisialisasi flask app
app = Flask(__name__)

# Memuat model terbaik
best_model = load_best_model('best_model.xgb')

# Route untuk ke halaman home
@app.route("/")
def home():
    return render_template("index.html")

# Route untuk menangani prediksi harga saham
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Mendapatkan input tanggal dari form HTML
            end_date_str = request.form['end_date']
            
            # Mengonversi string tanggal ke objek datetime
            end_date = pd.to_datetime(end_date_str)

            # Menentukan input_data (data historis harga saham)
            # Sebagai contoh, Anda dapat memuat data historis dari file CSV atau sumber data lainnya
            input_data = pd.read_csv('gt.csv')
            input_data['Date'] = pd.to_datetime(input_data['Date'])
            # Inisialisasi start_date
            start_date = pd.to_datetime('2023-12-02')

            input_data = input_data.drop(['Open', 'High', 'Low', 'Adj Close'], axis=1)

            # Menambahkan MA
            calculate_moving_averages(input_data)

            # Menambahkan RSI, Bollinger Bands, dan MACD
            input_data['RSI'] = calculate_rsi(input_data)
            #data['UpperBand'], data['MiddleBand'], data['LowerBand'] = calculate_bollinger_bands(data)
            input_data['MACD'], input_data['Signal_Line'] = calculate_macd(input_data)

            # Ekstraksi fitur-fitur waktu yang relevan
            input_data['Year'] = input_data['Date'].dt.year
            input_data['Month'] = input_data['Date'].dt.month
            input_data['Day'] = input_data['Date'].dt.day
            input_data['DayOfWeek'] = input_data['Date'].dt.dayofweek

            # Membagi Data menjadi Data Latih dan Data Uji
            X = input_data.drop(['Date', 'Close'], axis=1)
            y = input_data['Close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            y_pred = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            print(f'MAE: {mae}')

            # Proses input dan eksekusi prediksi
            predict_data = predict_future_with_technical_features(best_model, input_data, start_date, end_date)
            predict_data['Date'] = predict_data['Date'].astype(str)
            predict_data['Close(Predicted)'] = predict_data['Close(Predicted)'].astype(str)
            print("Nilai predict_data type:", predict_data.dtypes)
            print("Nilai predict_data:", predict_data)

            # Data untuk grafik
            graph_data_json = predict_data.to_json(orient='records')

            # Render template dan hasil prediksi
            return render_template("predict-result.html", predict_data=predict_data, graph_data=graph_data_json, mae=mae)
        except Exception as e:
                # Mencetak pesan kesalahan jika ada
                print("Error:", e)
                return "Error occurred. Please check the logs for details."

if __name__ == "__main__":
    app.run(debug=True)