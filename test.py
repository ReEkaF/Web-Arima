import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse
from io import BytesIO

app = Flask(__name__)

# Direktori untuk menyimpan file yang diupload
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Fungsi untuk mengecek ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fungsi untuk melatih model ARIMA
def train_arima_model(data):
    # Asumsi data dalam format yang benar, misalnya dataframe dengan satu kolom data waktu
    model = ARIMA(data, order=(1,1,2))  # Model ARIMA(1,1,2)
    model_fit = model.fit()
    return model_fit

# Fungsi untuk memprediksi menggunakan model ARIMA yang sudah dilatih
def predict(model_fit, steps=5):
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Fungsi untuk membuat grafik prediksi
def create_forecast_plot(data, forecast):
    plt.figure(figsize=(10,6))
    plt.plot(data, label='Data Historis')
    plt.plot(np.arange(len(data), len(data) + len(forecast)), forecast, color='red', label='Prediksi ARIMA')
    plt.title('Prediksi ARIMA')
    plt.xlabel('Waktu')
    plt.ylabel('Jumlah')
    plt.legend()

    # Menyimpan grafik ke dalam objek BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

# Halaman utama untuk upload file
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Baca data CSV
            data = pd.read_csv(file_path)

            # Pastikan data memiliki kolom yang valid untuk ARIMA (misalnya 'Jumlah')
            if 'Jumlah' not in data.columns:
                return "File harus memiliki kolom 'Jumlah'."

            # Melatih model ARIMA
            model_fit = train_arima_model(data['Jumlah'])

            # Memprediksi 5 langkah ke depan
            forecast = predict(model_fit, steps=5)

            # Membuat grafik prediksi
            img = create_forecast_plot(data['Jumlah'], forecast)

            return render_template('result.php', forecast=forecast, plot_url=img)
    return render_template('index.php')

# Halaman untuk menampilkan hasil prediksi
@app.route('/result', methods=['GET'])
def result():
    return render_template('result.php')

# Fungsi untuk mengirim gambar sebagai file untuk ditampilkan di php
@app.route('/plot.png')
def plot_png():
    data = pd.read_csv('uploads/wisatawan.csv')  # Sesuaikan nama file yang di-upload
    model_fit = train_arima_model(data['Jumlah'])
    forecast = predict(model_fit, steps=5)
    img = create_forecast_plot(data['Jumlah'], forecast)
    return send_file(img, mimetype='image/png')
# Route untuk menampilkan plot sebagai gambar PNG (jika diakses langsung)
@app.route('/plot.png')
def plot_png():
    file_path = 'uploads/wisatawan.csv'  # Sesuaikan nama file default
    data = pd.read_csv(file_path)

    bulan = int(request.args.get('bulan', 10))  # Default 5 bulan jika tidak ada input
    model_fit = train_arima_model(data['Jumlah'])
    forecast = predict(model_fit, steps=bulan)

    # Buat gambar plot dan simpan
    img_filename = create_forecast_plot(data['Jumlah'], forecast, bulan)

    img_path = os.path.join(app.config['IMAGE_FOLDER'], img_filename)
    return send_file(img_path, mimetype='image/png')
if __name__ == '__main__':
    app.run(debug=True)
