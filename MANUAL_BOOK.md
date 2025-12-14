# Buku Panduan Teknis & Penggunaan (Manual Book)
## Generalist Sentiment Analysis System

**Versi Dokumen:** 2.0  
**Terakhir Diperbarui:** 15 Desember 2025

---

## 1. Pendahuluan

### 1.1 Latar Belakang
Generalist Sentiment Analysis System dibangun untuk memenuhi kebutuhan analisis sentimen yang cepat, akurat, dan mudah digunakan terhadap teks berbahasa Indonesia. Sistem ini dirancang untuk menangani berbagai sumber data, mulai dari file internal (CSV/Excel) hingga data web real-time (Berita/Artikel), serta menyediakan fleksibilitas antara kecepatan (Machine Learning Klasik) dan akurasi kontekstual (Deep Learning).

### 1.2 Tujuan
*   Menyediakan platform terpadu untuk analisis opini publik.
*   Memfasilitasi pemantauan isu (*issue monitoring*) melalui pencarian web otomatis.
*   Memberikan wawasan (*insights*) berbasis data melalui visualisasi dan pengelompokan topik (*lustering*).

### 1.3 Target Pengguna
*   **Analisis Data:** Untuk memproses dataset besar.
*   **Humas/Public Relations:** Untuk memantau sentimen publik terhadap merek/isu.
*   **Pengembang/Administrator:** Untuk mengelola model dan infrastruktur sistem.

---

## 2. Spesifikasi Teknis Sistem

### 2.1 Arsitektur Teknologi (*Technology Stack*)

| Komponen | Teknologi | Keterangan |
| :--- | :--- | :--- |
| **Backend Framework** | Flask (Python 3.8+) | RESTful API, Jinja2 Templating |
| **Server** | Gunicorn | WSGI HTTP Server untuk Production |
| **Database** | SQLite (Dev) / PostgreSQL | Penyimpanan log analisis via SQLAlchemy |
| **Task Queue** | Redis & RQ | Manajemen antrean proses background (Training) |
| **Frontend** | HTML5, CSS3, Vanilla JS | Antarmuka pengguna responsif |
| **ML Libraries** | Scikit-Learn | Training model LinearSVC, Preprocessing |
| **DL Libraries** | PyTorch, Transformers | Inferensi model Roberta-base (Hugging Face) |
| **NLP Utilities** | Sastrawi, NLTK | Stemming & Tokenisasi Bahasa Indonesia |
| **Search Engine** | DuckDuckGo Search (ddgs) | Pencarian artikel web tanpa API key berbayar |

### 2.2 Struktur Proyek

```
generalist-sentiment-system/
├── app.py                 # Entry point aplikasi utama (Flask Config & Routes)
├── config.py              # Konfigurasi Environment (Dev/Prod)
├── worker.py              # Script worker untuk memproses Job Queue (Redis)
├── requirements.txt       # Daftar dependensi Python
├── modules/               # Modul logika bisnis
│   ├── analyzer.py        # Core Logic: Training, Prediksi, Clustering
│   ├── preprocessor.py    # Pipeline pembersihan teks
│   ├── dataset_finder.py  # Modul pencarian web
│   ├── tasks.py           # Fungsi yang dijalankan oleh background worker
│   ├── training_state.py  # Manajemen state proses training
│   ├── slang_dict.py      # Kamus normalisasi kata alay
│   └── lexicon_data.py    # Kamus kata positif/negatif dasar
├── models/                # Penyimpanan Model & Skema Database
│   ├── sentiment_log.py   # Definisi skema tabel database
│   └── (file .pkl)        # File model tersimpan (Pickle format)
├── templates/             # File HTML (Jinja2)
├── static/                # File CSS, JS, Gambar
└── instance/              # Database SQLite (generalist_sentiment.db)
```

### 2.3 Skema Database
Tabel utama `sentiment_logs` menyimpan riwayat setiap teks yang dianalisis.

| Kolom | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `id` | Integer (PK) | ID Unik log |
| `text` | Text | Teks asli yang dianalisis |
| `label` | String(50) | Hasil prediksi (positif/netral/negatif) |
| `sentiment_score` | Float | Skor polaritas (-1.0 s.d 1.0) |
| `confidence_score` | Float | Tingkat keyakinan model (0.0 s.d 1.0) |
| `cluster` | Integer | ID Klaster topik (0, 1, 2) |
| `source` | String(100) | Asal data (nama file atau URL) |
| `model_version` | String(50) | Versi model yang digunakan saat analisis |
| `metadata_json` | JSON | Data tambahan (judul, query pencarian, dll) |
| `created_at` | DateTime | Waktu analisis dilakukan |

---

## 3. Algoritma dan Metodologi

### 3.1 Pipeline Preprocessing Data
Data teks mentah melalui tahapan pembersihan ketat sebelum masuk ke model:
1.  **Case Folding:** Konversi huruf menjadi kecil (`.lower()`).
2.  **Cleaning Regex:** Menghapus angka dan karakter non-alfabet (`[^a-z\s]`).
3.  **Stopword Removal (Cerdas):** Menghapus kata hubung (stopword) menggunakan Sastrawi, namun **mengecualikan kata negasi** (tidak, bukan, jangan, belum, tak, kurang) agar makna kalimat tidak berbalik.
4.  **Slang Normalization:** Mengganti kata tidak baku (contoh: "gwbgt") menjadi baku ("gue banget") menggunakan kamus `modules/slang_dict.py`.
5.  **Stemming:** Mengubah kata berimbuhan menjadi kata dasar (contoh: "memakan" -> "makan") menggunakan algoritma Sastrawi dengan optimasi cache (`lru_cache`) untuk performa.

### 3.2 Model Analisis Sentimen

#### Mode Default: Linear Support Vector Classifier (LinearSVC)
*   **Algoritma:** Linear SVM dikenal memiliki performa tinggi pada data teks berdimensi tinggi.
*   **Feature Extraction:** TF-IDF Vectorizer (Unigram + Bigram) untuk menangkap konteks frasa pendek.
*   **Kalibrasi:** Menggunakan `CalibratedClassifierCV` (Sigmoid calibration) untuk mengubah output SVM (jarak margin) menjadi probabilitas (*Confidence Score*) yang dapat diinterpretasikan.
*   **Hyperparameter Tuning:** Menggunakan `GridSearchCV` untuk mencari nilai `C` (Regularization) terbaik dan strategi `class_weight`.

#### Mode Advanced: Transformers (Hugging Face)
*   **Model Base:** `w11wo/indonesian-roberta-base-sentiment-classifier`.
*   **Arsitektur:** RoBERTa (Robustly Optimized BERT Approach) yang telah dipra-latih pada korpus Bahasa Indonesia yang besar (CC-100-ID).
*   **Logika Inferensi:** Input token dibatasi maks 512 token. Output *logits* dinormalisasi menjadi label (Positif, Negatif, Netral).

### 3.3 Clustering Topik
*   **Algoritma:** K-Means Clustering.
*   **Jumlah Klaster (K):** 3 (Default).
*   **Input:** Vektor TF-IDF dari teks yang telah dipreprocess.
*   **Fungsi:** Mengelompokkan dokumen yang memiliki kesamaan kata-kata kunci ke dalam grup untuk memudahkan analisis tema dominan.

---

## 4. Panduan Penggunaan (User Manual)

### 4.1 Persiapan Data (Format File)
Untuk fitur **Upload File**, siapkan data dalam format:
1.  **Excel (.xlsx)** atau **CSV (.csv)**.
2.  File **WAJIB** memiliki setidaknya satu kolom yang berisi teks (sistem akan otomatis mendeteksi kolom bernama `text`, `content`, `review`, `ulasan`, atau `komentar`. Jika tidak ada, kolom pertama akan digunakan).

### 4.2 Melakukan Analisis Sentimen
1.  Buka halaman utama aplikasi.
2.  Pada bagian **Analisis File**:
    *   Klik "Browse" dan pilih file Anda.
    *   Pilih "Model Type": `Default` (Cepat) atau `Hugging Face` (Akurasi Tinggi, Lebih Lambat).
    *   Klik tombol **Analyze**.
3.  Tunggu proses selesai.
4.  Hasil akan muncul berupa:
    *   **Distribusi Sentimen:** Grafik Pie Chart.
    *   **Klaster Topik:** Grafik Batang jumlah data per klaster.
    *   **Tabel Data:** Pratinjau 100 data pertama beserta hasil prediksinya.

### 4.3 Pencarian & Analisis Web
1.  Pada bagian **Search & Analyze**:
    *   Ketikkan topik (misal: "Pemilu 2024").
    *   Klik tombol **Search**.
2.  Sistem akan mengambil data artikel dari DuckDuckGo, melakukan analisis, dan menampilkan hasilnya dalam format yang sama.

### 4.4 Melatih Model Baru (Training)
Fitur ini digunakan jika akurasi model standar dirasa kurang sesuai dengan domain data spesifik Anda.
1.  Siapkan file CSV dengan format kolom ketat: `text` dan `label` (isi label harus: positif / negatif / netral).
2.  Kirim file ke kolom upload Training (biasanya via endpoint API atau UI admin jika diaktifkan).
3.  Proses berjalan di latar belakang (*Background Job*).
4.  Model baru akan menggantikan model lama setelah selesai dan tervalidasi.

---

## 5. Panduan Pengembang & Deployment

### 5.1 Persiapan Lingkungan (Installation)
Pastikan Python 3.8+ dan Redis terinstall.

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/your-repo/generalist-sentiment.git
    cd generalist-sentiment
    ```
2.  **Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # venv\Scripts\activate   # Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Database & Redis:**
    *   Pastikan Redis service berjalan (`redis-server`).
    *   Database SQLite akan dibuat otomatis saat aplikasi dijalankan pertama kali.

### 5.2 Menjalankan Aplikasi
Aplikasi membutuhkan dua proses terminal terpisah:

**Terminal 1 (Worker Processor):**
```bash
python worker.py
```
*Worker bertugas menangani request training yang berat.*

**Terminal 2 (Web Server):**
```bash
python app.py
```
*Aplikasi berjalan di `http://localhost:5000`.*

### 5.3 Dokumentasi API Endpoint
Akses Swagger UI di `/apidocs` (jika diaktifkan) atau gunakan referensi berikut:

**1. POST `/analyze`**
*   **Deskripsi:** Menganalisis file upload.
*   **Body (Multipart/Form-Data):**
    *   `file`: File dataset (.csv/.xlsx).
    *   `model_type`: 'default' | 'hf'.
*   **Response (200 OK):** JSON object berisi statistik dan data preview.

**2. POST `/search_and_analyze`**
*   **Deskripsi:** Cari & analisis web.
*   **Body (JSON):**
    *   `query`: "Kata kunci pencarian".
    *   `limit`: (Int) Maksimum hasil (Default 100).
    *   `model_type`: 'default' | 'hf'.
*   **Response (200 OK):** JSON object hasil analisis.

**3. POST `/train`**
*   **Deskripsi:** Memulai training model baru.
*   **Body (Multipart/Form-Data):**
    *   `file`: File training (Columns: text, label).
*   **Response (200 OK):** `{"job_id": "...", "status": "started"}`.

### 5.4 Troubleshooting
*   **Error: "Redis Connection Error"**: Pastikan service redis menyala. Cek `REDIS_URL` di `config.py` atau environment variable.
*   **Training Tidak Selesai-Selesai**: Cek log pada terminal `worker.py`. Kemungkinan terjadi error pada dataset (format salah/kosong).
*   **Akurasi Rendah**: Coba gunakan dataset training yang lebih besar dan seimbang antar kelas sentimennya.
*   **Memory Error (OOM)**: Saat menggunakan model Hugging Face, memori bisa cepat habis. Kurangi batch size atau gunakan mode Default (LinearSVC) pada mesin dengan RAM terbatas.

---
*Dokumen ini bersifat rahasia dan diperuntukkan bagi penggunaan internal tim pengembang dan pengguna sistem.*
