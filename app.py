import os
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
import uuid
from flask import send_from_directory


app = Flask(__name__, static_folder='static')  # Menyajikan folder statis dari folder 'static'

app.secret_key = 'admin'  # Ganti dengan secret key yang aman
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
IMAGE_FOLDER = 'images'  # Folder untuk menyimpan gambar plot
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Fungsi untuk mengecek ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fungsi untuk melatih model ARIMA
def train_arima_model(data):
    model = ARIMA(data, order=(1,1,2))  # Model ARIMA(1,1,2)
    model_fit = model.fit()
    return model_fit

# Fungsi untuk memprediksi menggunakan model ARIMA yang sudah dilatih
def predict(model_fit, steps):
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Fungsi untuk membuat grafik prediksi dan menyimpannya ke file
def create_forecast_plot(data, forecast, bulan):
    plt.figure(figsize=(10,6))
    plt.plot(data, label='Data Historis', color='blue')
    plt.plot(np.arange(len(data), len(data) + len(forecast)), forecast, color='red', label='Prediksi ARIMA')
    plt.title('Prediksi ARIMA')
    plt.xlabel('Waktu')
    plt.ylabel('Jumlah')
    plt.legend()

    # Menyimpan grafik ke dalam folder images
    img_filename = f'forecast.jpg'  # Nama file unik dengan UUID
    img_path = os.path.join(app.config['IMAGE_FOLDER'], img_filename)
    plt.savefig(img_path)
    plt.close()  # Menutup plot untuk menghindari memory leak
    return img_filename  # Mengembalikan nama file gambar

# Dummy data untuk admin, bisa diganti dengan data yang diambil dari database
admins = {
    'admin': {'password': generate_password_hash('admin123')}  # Gantilah password ini dengan hash password yang aman
}

# Halaman login
@app.route('/', methods=['GET', 'POST'])
def login():
    if 'admin' in session:
        return redirect(url_for('admin_dashboard'))

    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Validasi username dan password
        if username in admins:
            admin = admins[username]
            if check_password_hash(admin['password'], password):  # Menggunakan hash password yang lebih aman
                session['admin'] = username
                return redirect(url_for('admin_dashboard'))
            else:
                error = 'Password salah'
        else:
            error = 'Username tidak ditemukan'
    
    return render_template('login.html', error=error)

# Halaman dashboard admin
@app.route('/admin/index')
def admin_dashboard():
    if 'admin' not in session:
        return redirect(url_for('login'))
    return render_template('admin/index.html')

# Halaman ARIMA
@app.route('/admin/Arima', methods=['GET', 'POST'])
def admin_arima():
    if 'admin' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        bulan = request.form.get('bulan', type=int)  # Mengambil input bulan sebagai integer
        
        if not file or file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Baca data CSV
            data = pd.read_csv(file_path)

            # Validasi kolom data
            if 'Jumlah' not in data.columns:
                return "File harus memiliki kolom 'Jumlah'."

            # Melatih model ARIMA
            model_fit = train_arima_model(data['Jumlah'])

            # Prediksi sesuai jumlah bulan yang diminta
            forecast = predict(model_fit, steps=bulan)

            # Buat grafik prediksi dan simpan ke folder images
            img_filename = create_forecast_plot(data['Jumlah'], forecast, bulan)

            # Render hasil ke template dengan mengirimkan URL gambar
            img_url = url_for('static', filename=f'images/{img_filename}')
            return render_template('admin/result.html', forecast=forecast, plot_url=img_url)
    
    return render_template('admin/arima.html')
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

# Route untuk logout
@app.route('/logout')
def logout():
    session.pop('admin', None)  # Menghapus session admin
    return redirect(url_for('login'))
# Run aplikasi
if __name__ == '__main__':
    app.run(debug=True)
