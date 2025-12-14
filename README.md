# Generalist Sentiment Analysis System

Sistem analisis sentimen berbasis web yang dirancang untuk memproses ulasan berbahasa Indonesia. Aplikasi ini bersifat "Generalist", artinya dapat dilatih dengan berbagai jenis dataset ulasan (produk, layanan, film, dll) untuk memprediksi sentimen dan mengelompokkan topik secara otomatis.

## 🚀 Fitur Utama

-   **Analisis Sentimen (Supervised)**: Mengklasifikasikan teks ke dalam kategori sentimen (misal: Positif/Negatif) menggunakan algoritma **Naive Bayes**.
-   **Topic Clustering (Unsupervised)**: Mengelompokkan ulasan-ulasan yang memiliki kemiripan topik menggunakan algoritma **K-Means Clustering**, tanpa memerlukan label sebelumnya.
-   **Preprocessing Bahasa Indonesia**: Pembersihan teks yang optimal menggunakan **Sastrawi** (Stemming & Stopword Removal) yang telah dioptimasi dengan caching.
-   **Custom Training**: Pengguna dapat melatih ulang model dengan dataset mereka sendiri melalui antarmuka web.
-   **Visualisasi Data**: Menampilkan grafik distribusi sentimen dan hasil clustering.

## 🛠️ Teknologi

-   **Backend**: Python (Flask)
-   **Machine Learning**: Scikit-Learn (MultinomialNB, KMeans, TF-IDF)
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

## 🖥️ Cara Penggunaan

1.  **Jalankan Aplikasi**:
    ```bash
    python app.py
    ```
2.  Buka browser dan akses alamat: `http://localhost:5000`.

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

```
generalist-sentiment-system/
├── app.py                  # Entry point aplikasi Flask
├── modules/
│   ├── analyzer.py         # Logika ML (Naive Bayes & KMeans)
│   └── preprocessor.py     # Logika Preprocessing (Sastrawi)
├── static/                 # CSS, JS, Image
├── templates/              # File HTML
├── requirements.txt        # Daftar library Python
└── README.md               # Dokumentasi
```

## 📝 Catatan Penting
-   **Performa**: Proses stemming (pemotongan kata dasar) menggunakan Sastrawi bisa memakan waktu untuk dataset yang sangat besar. Sistem ini sudah dilengkapi dengan mekanisme **Caching** untuk mempercepat proses pada kata-kata yang berulang.
-   **Model**: Model disimpan dalam format `.pkl` (Pickle). Pastikan versi library scikit-learn konsisten jika memindahkan model antar mesin.
