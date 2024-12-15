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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
def create_acf_plot(residuals):
    fig_acf_residual, ax_acf_residual = plt.subplots(figsize=(10, 6))
    plot_acf(residuals, ax=ax_acf_residual)
    acf_filename = f'acf_{uuid.uuid4().hex}.jpg'  # Nama file unik dengan UUID
    acf_path = os.path.join(app.config['IMAGE_FOLDER'], acf_filename)
    fig_acf_residual.savefig(acf_path)
    plt.close(fig_acf_residual)  # Menutup plot untuk menghindari memory leak
    return acf_filename  # Mengembalikan nama file gambar ACF

# Fungsi untuk membuat dan menyimpan plot PACF
def create_pacf_plot(residuals):
    fig_pacf_residual, ax_pacf_residual = plt.subplots(figsize=(10, 6))
    plot_pacf(residuals, ax=ax_pacf_residual, method='ywm')
    pacf_filename = f'pacf_{uuid.uuid4().hex}.jpg'  # Nama file unik dengan UUID
    pacf_path = os.path.join(app.config['IMAGE_FOLDER'], pacf_filename)
    fig_pacf_residual.savefig(pacf_path)
    plt.close(fig_pacf_residual)  # Menutup plot untuk menghindari memory leak
    return pacf_filename  # Mengembalikan nama file gambar PAC
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

@app.route('/admin/data')
def admin_data():
    if 'admin' not in session:
        return redirect(url_for('login'))
    data = pd.read_csv('uploads/wisatawan.csv')
    data_records = data.to_dict(orient='records') 
    return render_template('admin/data.html', data=data_records)


@app.route('/admin/data/delete', methods=['POST'])
def admin_data_delete():
    if request.method == 'POST':
        # Retrieve the index from the form
        try:
            index_to_delete = int(request.form.get('index')) - 1  # Adjust index for 0-based index
        except ValueError:
            return "Invalid index. Please enter a valid number.", 400

        # Load the CSV data
        data = pd.read_csv('uploads/wisatawan.csv')

        # Check if the index is within the bounds of the data
        if index_to_delete < 0 or index_to_delete >= len(data):
            return "Index out of range. Please provide a valid index.", 400

        # Drop the row with the specified index
        data = data.drop(index_to_delete)

        # Save the updated data back to the CSV file
        data.to_csv('uploads/wisatawan.csv', index=False)

        # Convert the data to a dictionary to pass to the template
        data_records = data.to_dict(orient='records')

        return render_template('admin/data.html', data=data_records)

@app.route('/admin/data/add', methods=['POST'])
def admin_data_add():
    if request.method == 'POST':
        # Ambil data dari form
        Date = request.form.get('Date')
        Jumlah = request.form.get('Jumlah')

        # Validasi input
        if not Date or not Jumlah:
            return "Semua field harus diisi!", 400

        # Validasi format Date (YYYY-MM)
        try:
            # Mengonversi ke format tanggal untuk memastikan formatnya benar
            New_Date=pd.to_datetime(Date, format='%Y-%m').strftime('%Y-%m')
        except ValueError:
            return "Format Date salah. Gunakan format YYYY-MM.", 400

        # Validasi 'Jumlah' sebagai angka
        try:
            # Menghapus karakter selain angka (contoh: titik atau koma)
            Jumlah = int(Jumlah.replace('.', '').replace(',', ''))
        except ValueError:
            return "Jumlah Wisatawan harus berupa angka valid.", 400

        # Siapkan data baru
        new_data = pd.DataFrame([{'Date': New_Date, 'Jumlah': Jumlah}])

        # Coba membaca data CSV, jika tidak ada file maka buat file baru
        try:
            data = pd.read_csv('uploads/wisatawan.csv')
        except FileNotFoundError:
            data = pd.DataFrame(columns=['Date', 'Jumlah'])

        # Cek jika data dengan tanggal yang sama sudah ada
        if Date in data['Date'].values:
            return "Data dengan tanggal ini sudah ada!", 400

        # Gabungkan data lama dengan data baru
        data = pd.concat([data, new_data], ignore_index=True)
        data = data.sort_values(by='Date').reset_index(drop=True)  # urutkan berdasarkan 'Date'

        # Simpan kembali data ke CSV
        try:
            data.to_csv('uploads/wisatawan.csv', index=False)
        except Exception as e:
            return f"Terjadi kesalahan saat menyimpan data ke CSV: {str(e)}", 500

        # Konversi data menjadi dictionary dan render ulang halaman
        data_records = data.to_dict(orient='records')
        return render_template('admin/data.html', data=data_records)
    
@app.route('/admin/Arima_Manual_parameter', methods=['GET', 'POST'])
def admin_arima_manual_parameter():
    if 'admin' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Membaca data dari file CSV
            data = pd.read_csv('uploads/wisatawan.csv')
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m')  # Mengonversi kolom Date ke datetime
            data.set_index('Date', inplace=True)
            data = data['Jumlah']  # Mengambil hanya kolom 'Jumlah'

            # Mengambil parameter p, d, q dari form
            p = request.form.get('p', type=int)
            d = request.form.get('d', type=int)
            q = request.form.get('q', type=int)
            bulan = request.form.get('bulan', type=int)
            steps = bulan  # Jumlah langkah ramalan

            # Membangun dan melatih model ARIMA
            model = ARIMA(data.values, order=(p, d, q))
            model_fit = model.fit()

            # Melakukan peramalan
            forecast = model_fit.forecast(steps=steps)
            last_period = data.index[-1] + pd.DateOffset(months=1)  # Menambahkan satu bulan ke periode terakhir
            forecast_index = pd.date_range(start=last_period, periods=steps, freq='M')  # Membuat index untuk peramalan

            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Jumlah'])

            # Menghitung nilai metrik evaluasi
            length = min(len(data), len(forecast_df))
            actual = data.values[-length:]  # Menggunakan .values untuk mendapatkan array numpy
            predicted = forecast[:length]  # Prediksi sesuai panjang data aktual

            if len(actual) > 0:
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                mae = np.mean(np.abs(actual - predicted))
                mse = np.mean((actual - predicted) ** 2)
                rmse = np.sqrt(mse)
            else:
                return "Data kurang", 400

            # Plot hasil peramalan
            plt.figure(figsize=(10, 6))
            plt.plot(data, label="Data Asli")
            plt.plot(forecast_df, label="Peramalan", color='red')
            plt.legend(loc="best")
            plt.title("Peramalan Waktu Menggunakan ARIMA")
            plt.show()

            # Plot ACF dan PACF dari residuals
            residuals = model_fit.resid
            fig_acf_residual, ax_acf_residual = plt.subplots(figsize=(10, 6))
            plot_acf(residuals, ax=ax_acf_residual)
            plt.show()

            fig_pacf_residual, ax_pacf_residual = plt.subplots(figsize=(10, 6))
            plot_pacf(residuals, ax=ax_pacf_residual, method='ywm')
            plt.show()

        except Exception as e:
            return f"Terjadi kesalahan saat melakukan peramalan: {str(e)}", 500

    return render_template('admin/arima_manual_parameter.html')

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
