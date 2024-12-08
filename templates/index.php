<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload CSV untuk ARIMA</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <h1>Upload File CSV untuk Model ARIMA</h1>
    <form method="POST" enctype="multipart/form-data">
        <label for="file">Pilih File CSV:</label>
        <input type="file" name="file" accept=".csv" required>
        <br><br>
        <button type="submit">Upload dan Prediksi</button>
    </form>
</body>
</html>
