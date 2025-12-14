# Generalist Sentiment Analysis System

Sistem analisis sentimen berbasis web yang dirancang untuk memproses ulasan berbahasa Indonesia. Aplikasi ini bersifat "Generalist", artinya dapat dilatih dengan berbagai jenis dataset ulasan (produk, layanan, film, dll) untuk memprediksi sentimen dan mengelompokkan topik secara otomatis.

## 🚀 Fitur Utama

-   **Analisis Sentimen (Hybrid)**: Mendukung dua mode analisis:
    -   **Classic**: Naive Bayes (Cepat, efisien).
    -   **Advanced**: Hugging Face Transformers (Deep Learning, akurasi tinggi, konteks lebih baik).
-   **Topic Clustering (Unsupervised)**: Mengelompokkan ulasan-ulasan yang memiliki kemiripan topik menggunakan algoritma **K-Means Clustering**.
-   **Asynchronous Training**: Proses training model berjalan di latar belakang menggunakan **Redis Queue & Worker**, sehingga tidak memblokir antarmuka pengguna.
-   **Database Persistence**: Menyimpan riwayat analisis dan meta-data model menggunakan **SQLite**.
-   **API Documentation**: Dokumentasi API interaktif yang lengkap menggunakan **Swagger UI**.
-   **Preprocessing Bahasa Indonesia**: Pembersihan teks yang optimal menggunakan **Sastrawi**.
-   **Custom Training**: Pengguna dapat melatih ulang model dengan dataset mereka sendiri.
-   **Visualisasi Data**: Menampilkan grafik distribusi sentimen dan hasil clustering.

## 🛠️ Teknologi

-   **Backend**: Python (Flask)
-   **Machine Learning**: Scikit-Learn, Transformers (Hugging Face)
-   **Task Queue**: Redis & RQ
-   **Database**: SQLite (SQLAlchemy)
-   **Documentation**: Flasgger (Swagger/OpenAPI)
-   **NLP**: Sastrawi & Regex
-   **Frontend**: HTML5, CSS3, Vanilla JS
-   **Data Processing**: Pandas

## 📦 Instalasi

1.  **Clone Repository** (atau ekstrak source code).
2.  **Siapkan Environment Python**:
    Disarankan menggunakan virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/Mac
    # venv\Scripts\activate   # Untuk Windows
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install & Start Redis** (Wajib untuk Queue):
    -   **Linux (Ubuntu/Debian)**:
        ```bash
        sudo apt-get install redis-server
        sudo service redis-server start
        ```
    -   **Windows**: Gunakan WSL atau download Redis installer.
    -   **Mac**: `brew install redis` && `brew services start redis`

## 🖥️ Cara Penggunaan

1.  **Jalankan Aplikasi**:
    ```bash
    # Terminal 1: Jalankan Web Server
    python app.py

    # Terminal 2: Jalankan Worker (untuk background training)
    python worker.py
    ```
2.  Buka browser dan akses alamat: `http://localhost:5000`.
3.  Untuk melihat **Dokumentasi API**, buka `http://localhost:5000/apidocs`.

### Alur Kerja

#### 1. Training Model (Opsional tapi Disarankan)
Jika baru pertama kali dijalankan, model mungkin belum tersedia.
-   Masuk ke tab **Training Model**.
-   Upload file **CSV** atau **Excel** (.xlsx).
-   **Format Wajib**: Harus memiliki kolom `text` (isi ulasan) dan `label` (klasifikasi sentimen).
-   Klik **Mulai Training**. Model akan disimpan sebagai `model_sentiment.pkl`.

#### 2. Analisis Data
-   Masuk ke tab **Analisis Data**.
-   Upload file ulasan (CSV/Excel). Kolom teks akan dideteksi secara otomatis.
-   Sistem akan memproses data dan menampilkan:
    -   Statistik Sentimen.
    -   Grafik Distribusi.
    -   Tabel hasil prediksi per ulasan.

## 📂 Struktur Proyek

generalist-sentiment-system/
├── app.py                  # Entry point aplikasi Flask
├── data/
│   └── samples/            # Contoh dataset (CSV)
├── models/                 # Direktori model tersimpan (PKL)
├── modules/
│   ├── analyzer.py         # Logika ML (Naive Bayes & KMeans)
│   └── preprocessor.py     # Logika Preprocessing (Sastrawi)
├── scripts/                # Script utilitas dan verifikasi
├── static/                 # CSS, JS, Image
├── templates/              # File HTML
├── requirements.txt        # Daftar library Python
└── README.md               # Dokumentasi

## 📝 Catatan Penting
-   **Performa**: Proses stemming (pemotongan kata dasar) menggunakan Sastrawi bisa memakan waktu untuk dataset yang sangat besar. Sistem ini sudah dilengkapi dengan mekanisme **Caching** untuk mempercepat proses pada kata-kata yang berulang.
-   **Model**: Model disimpan dalam format `.pkl` (Pickle). Pastikan versi library scikit-learn konsisten jika memindahkan model antar mesin.
