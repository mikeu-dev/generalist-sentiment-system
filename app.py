from flask import Flask, render_template, request, jsonify
import os
import threading
import pandas as pd
import logging
from logging.config import dictConfig
from datetime import datetime
from config import DevelopmentConfig, ProductionConfig
from modules.preprocessor import TextPreprocessor
from modules.analyzer import SentimentAnalyzer
from models.sentiment_log import db, SentimentLog
from flasgger import Swagger
from rq import Queue
from redis import Redis
from modules.tasks import run_training_background
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from utils.security import (
    validate_and_secure_filename,
    validate_file_size,
    sanitize_query_string,
    validate_limit_parameter,
    calculate_file_hash
)

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
swagger = Swagger(app)

# Koneksi Redis
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
redis_conn = Redis.from_url(redis_url)
queue = Queue(connection=redis_conn)

# Rate Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=app.config.get('RATELIMIT_STORAGE_URL', redis_url)
)

# Memuat Konfigurasi
if os.environ.get('FLASK_ENV') == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)

db.init_app(app)
with app.app_context():
    db.create_all()


# Pastikan direktori upload tersedia
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inisialisasi modul
logger.info("Memuat modul...")
preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer()
logger.info("Modul berhasil dimuat.")

# Register monitoring routes
from utils.monitoring import register_monitoring_routes
register_monitoring_routes(app, db, redis_conn, analyzer, SentimentLog)

@app.route('/')
def index():
    return render_template('index.html', model_trained=analyzer.is_trained)

# Manajer Status Training
from modules.training_state import TrainingStateManager

# Inisialisasi Manajer Status
# Gunakan UPLOAD_FOLDER agar file persistent dan accessible
state_manager = TrainingStateManager(upload_folder=app.config['UPLOAD_FOLDER'])

# run_training_background dipindahkan ke modules/tasks.py



@app.route('/train', methods=['POST'])
@limiter.limit("3 per hour")  # Training sangat resource-intensive
def train():
    """
    Melatih model Naive Bayes baru.
    ---
    tags:
      - Training
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Dataset dengan kolom 'text' dan 'label'.
    responses:
      200:
        description: Proses training berhasil dimulai.
    """
    try:
        current_status = state_manager.get_status()
        if current_status["is_training"]:
            return jsonify({"error": "Training sedang berjalan. Harap tunggu."}), 409

        if 'file' not in request.files:
            return jsonify({"error": "Tidak ada bagian file"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Tidak ada file yang dipilih"}), 400
        
        # Validasi filename dan extension
        is_valid, safe_filename, error_msg = validate_and_secure_filename(file.filename)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        # Validasi ukuran file
        is_valid_size, size_error = validate_file_size(file)
        if not is_valid_size:
            return jsonify({"error": size_error}), 400
        
        # Simpan file dengan nama yang aman
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        # Calculate hash untuk tracking
        file_hash = calculate_file_hash(filepath)
        logger.info(f"Training file uploaded: {safe_filename}, hash: {file_hash}")
        
        # Mulai pekerjaan di latar belakang via RQ
        job = queue.enqueue(run_training_background, filepath, app.config['UPLOAD_FOLDER'])
        logger.info(f"Training job enqueued: {job.id}")
        
        return jsonify({
            "message": "Proses training dimulai di latar belakang (Queue).",
            "status": "started",
            "job_id": job.id
        })
    
    except Exception as e:
        logger.error(f"Error in training endpoint: {e}", exc_info=True)
        return jsonify({"error": "Terjadi kesalahan saat memulai training"}), 500

@app.route('/train_status', methods=['GET'])
def get_train_status():
    return jsonify(state_manager.get_status())

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")  # Limit analisis untuk mencegah abuse
def analyze():
    """
    Analisis sentimen dari file yang diunggah.
    ---
    tags:
      - Analisis
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: File CSV atau Excel yang berisi kolom teks.
      - name: model_type
        in: formData
        type: string
        enum: ['default', 'hf']
        default: 'default'
        description: Pilih 'default' (Naive Bayes) atau 'hf' (Hugging Face / Deep Learning).
    responses:
      200:
        description: Hasil analisis termasuk distribusi, klaster, dan pratinjau data.
      400:
        description: Input tidak valid atau format file salah.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Tidak ada bagian file"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Tidak ada file yang dipilih"}), 400
        
        # Validasi filename dan extension
        is_valid, safe_filename, error_msg = validate_and_secure_filename(file.filename)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        # Validasi ukuran file
        is_valid_size, size_error = validate_file_size(file)
        if not is_valid_size:
            return jsonify({"error": size_error}), 400
        
        # Simpan file dengan nama yang aman
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        # Cek tipe model
        model_type = request.form.get('model_type', 'default')
        use_hf = (model_type == 'hf')

        # Baca file
        if filepath.endswith('.csv'):
            try:
                df = pd.read_csv(filepath)
            except Exception:
                logger.warning("Pembacaan CSV standar gagal, mencoba mode robust...")
                df = pd.read_csv(filepath, engine='python', on_bad_lines='skip', quotechar='"', encoding_errors='ignore')
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            return jsonify({"error": "Format file tidak didukung."}), 400
        
        # Validasi dataframe tidak kosong
        if df.empty:
            return jsonify({"error": "File tidak berisi data"}), 400
        
        # Mendeteksi nama kolom teks secara fleksibel
        text_col = None
        for col in ['text', 'content', 'review', 'ulasan', 'komentar']:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col:
            text_col = df.columns[0]
        
        raw_texts = df[text_col].astype(str).tolist()
        
        # Validasi jumlah data
        if len(raw_texts) == 0:
            return jsonify({"error": "Tidak ada teks yang ditemukan dalam file"}), 400
        
        if len(raw_texts) > 10000:
            return jsonify({"error": "Terlalu banyak data. Maksimum 10,000 baris"}), 400
        
        logger.info(f"Melakukan preprocessing data untuk analisis... ({len(raw_texts)} rows)")
        clean_texts = preprocessor.preprocess_batch(raw_texts)
        
        results = {"total": len(raw_texts), "distribution": {}, "clusters": []}
        
        # Prediksi Sentimen & Clustering
        logger.info(f"Memprediksi sentimen secara detail... Gunakan HF: {use_hf}")
        details = analyzer.predict_detailed(clean_texts, use_hf=use_hf)
        
        # Clustering Topik
        logger.info("Mengelompokkan topik (Clustering)...")
        clusters = analyzer.cluster_topics(clean_texts, n_clusters=3)
        
        # Simpan ke DB dan siapkan respons
        preview_data = []
        distribution = {}
        
        for i, text in enumerate(raw_texts):
            detail = details[i]
            cluster_id = int(clusters[i]) if i < len(clusters) else 0
            
            # Log Database
            log = SentimentLog(
                text=text,
                label=detail['label'],
                sentiment_score=detail['sentiment_score'],
                confidence_score=detail['confidence_score'],
                cluster=cluster_id,
                source='file_upload',
                metadata_json={'filename': safe_filename},
                model_version=detail['model_version']
            )
            db.session.add(log)
            
            # Data Respons
            if i < 100:
                preview_data.append({
                    "text": text,
                    "sentiment": detail['label'],
                    "cluster": cluster_id,
                    "score": detail['sentiment_score'],
                    "source": safe_filename,
                    "title": "Unggah File",
                    "preprocessed": clean_texts[i]
                })
            
            # Statistik Distribusi
            lbl = detail['label']
            distribution[lbl] = distribution.get(lbl, 0) + 1
        
        db.session.commit()
        
        # Statistik Klaster
        cluster_counts = {}
        for c in clusters:
            c_int = int(c)
            cluster_counts[c_int] = cluster_counts.get(c_int, 0) + 1

        results = {
            "total": len(raw_texts),
            "distribution": distribution,
            "cluster_counts": cluster_counts,
            "data": preview_data,
            "model_version": details[0]['model_version'] if details else "unknown"
        }
        
        if not analyzer.is_trained and details and details[0]['model_version'] == 'lexicon_rule_based':
             results['warning'] = "Model belum dilatih. Menggunakan pendekatan berbasis aturan (Lexicon)."

        return jsonify(results)

    except pd.errors.EmptyDataError:
        return jsonify({"error": "File CSV kosong atau tidak valid"}), 400
    except pd.errors.ParserError:
        return jsonify({"error": "Format CSV tidak valid"}), 400
    except ValueError as e:
        logger.error(f"Validation error in analyze: {e}")
        return jsonify({"error": "Data tidak valid"}), 400
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in analyze endpoint: {e}", exc_info=True)
        return jsonify({"error": "Terjadi kesalahan saat menganalisis data"}), 500

from modules.dataset_finder import DatasetFinder

# Inisialisasi DatasetFinder
dataset_finder = DatasetFinder()

@app.route('/search_and_analyze', methods=['POST'])
@limiter.limit("5 per minute")  # Limit search lebih ketat karena hit external API
def search_and_analyze():
    """
    Mencari data dari web dan menganalisis sentimen hasilnya.
    ---
    tags:
      - Analisis
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            query:
              type: string
              example: "Pemilu 2024"
            model_type:
              type: string
              enum: ['default', 'hf']
            limit:
              type: integer
              example: 100
    responses:
      200:
        description: Hasil pencarian dan analisis sentimen.
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Query diperlukan"}), 400
        
        # Sanitasi dan validasi query
        is_valid, sanitized_query, error_msg = sanitize_query_string(data['query'])
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        logger.info(f"Menerima query pencarian: {sanitized_query}")
        
        # Validasi limit parameter
        limit_val = data.get('limit', 100)
        validated_limit, limit_warning = validate_limit_parameter(limit_val, min_val=1, max_val=500, default=100)
        
        if limit_warning:
            logger.warning(limit_warning)
        
        # 1. Pencarian
        logger.info("Mencari di web...")
        raw_texts = dataset_finder.search(sanitized_query, max_results=validated_limit)
        
        model_type = data.get('model_type', 'default')
        use_hf = (model_type == 'hf')
        
        if not raw_texts:
            return jsonify({"error": "Tidak ada data ditemukan untuk topik tersebut."}), 404
        
        # Validasi jumlah hasil
        if len(raw_texts) > 500:
            logger.warning(f"Too many results ({len(raw_texts)}), truncating to 500")
            raw_texts = raw_texts[:500]
            
        # 2. Preprocessing
        logger.info(f"Preprocessing hasil pencarian... ({len(raw_texts)} items)")
        texts_only = [r['text'] for r in raw_texts]
        clean_texts = preprocessor.preprocess_batch(texts_only)
        
        results = {
            "query": sanitized_query,
            "total": len(raw_texts), 
            "distribution": {}, 
            "clusters": []
        }
        
        # 3. Prediksi & Klaster & Simpan
        details = analyzer.predict_detailed(clean_texts, use_hf=use_hf)
        clusters = analyzer.cluster_topics(clean_texts, n_clusters=3)
        
        preview_data = []
        distribution = {}
        cluster_counts = {}
        
        for i, item in enumerate(raw_texts):
            text = item['text']
            source = item['source']
            title = item.get('title', 'Tanpa Judul')
            
            detail = details[i]
            cluster_id = int(clusters[i]) if i < len(clusters) else 0
            
            # Log Database
            log = SentimentLog(
                text=text,
                label=detail['label'],
                sentiment_score=detail['sentiment_score'],
                confidence_score=detail['confidence_score'],
                cluster=cluster_id,
                source=source,
                metadata_json={'query': sanitized_query, 'title': title},
                model_version=detail['model_version']
            )
            db.session.add(log)
            
            # Data Respons
            if i < 100:
                preview_data.append({
                    "text": text,
                    "sentiment": detail['label'],
                    "cluster": cluster_id,
                    "source": source,
                    "title": title,
                    "preprocessed": clean_texts[i]
                })
            
            # Statistik
            lbl = detail['label']
            distribution[lbl] = distribution.get(lbl, 0) + 1
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            
        db.session.commit()
        
        results['distribution'] = distribution
        results['cluster_counts'] = cluster_counts
        results['data'] = preview_data
        results['method'] = "model_prediction" if analyzer.is_trained else "lexicon_fallback"
        
        if limit_warning:
            results['warning'] = limit_warning
            
        return jsonify(results)

    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        return jsonify({"error": "Data request tidak lengkap"}), 400
    except ValueError as e:
        logger.error(f"Validation error in search: {e}")
        return jsonify({"error": "Parameter tidak valid"}), 400
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in search_and_analyze endpoint: {e}", exc_info=True)
        return jsonify({"error": "Terjadi kesalahan saat mencari dan menganalisis data"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
