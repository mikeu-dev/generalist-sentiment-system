from flask import Flask, render_template, request, jsonify
import os
import threading
import pandas as pd
import logging
from logging.config import dictConfig
from config import DevelopmentConfig, ProductionConfig
from modules.preprocessor import TextPreprocessor
from modules.analyzer import SentimentAnalyzer
from models.sentiment_log import db, SentimentLog
from flasgger import Swagger
from rq import Queue
from redis import Redis
from modules.tasks import run_training_background

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
swagger = Swagger(app)

# Koneksi Redis
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
redis_conn = Redis.from_url(redis_url)
queue = Queue(connection=redis_conn)

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
    current_status = state_manager.get_status()
    if current_status["is_training"]:
        return jsonify({"error": "Training sedang berjalan. Harap tunggu."}), 409

    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada bagian file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Mulai pekerjaan di latar belakang via RQ
        job = queue.enqueue(run_training_background, filepath, app.config['UPLOAD_FOLDER'])
        logger.info(f"Training job antrean: {job.id}")
        
        return jsonify({
            "message": "Proses training dimulai di latar belakang (Queue).",
            "status": "started",
            "job_id": job.id
        })

@app.route('/train_status', methods=['GET'])
def get_train_status():
    return jsonify(state_manager.get_status())

@app.route('/analyze', methods=['POST'])
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
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada bagian file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Cek tipe model
        model_type = request.form.get('model_type', 'default')
        use_hf = (model_type == 'hf')

        try:
            if filepath.endswith('.csv'):
                try:
                    df = pd.read_csv(filepath)
                except Exception:
                    # Fallback untuk error parsing (misal: kesalahan tanda kutip)
                    logger.warning("Pembacaan CSV standar gagal, mencoba mode robust...")
                    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip', quotechar='"', encoding_errors='ignore')
            elif filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                return jsonify({"error": "Format file tidak didukung."}), 400
            
            # Mendeteksi nama kolom teks secara fleksibel
            text_col = None
            for col in ['text', 'content', 'review', 'ulasan', 'komentar']:
                if col in df.columns:
                    text_col = col
                    break
            
            if not text_col:
                # Jika tidak ada kolom yang dikenal, pilih kolom objek/string pertama atau kolom paling awal
                text_col = df.columns[0]
            
            raw_texts = df[text_col].astype(str).tolist()
            
            logger.info("Melakukan preprocessing data untuk analisis...")
            clean_texts = preprocessor.preprocess_batch(raw_texts)
            
            results = {"total": len(raw_texts), "distribution": {}, "clusters": []}
            
            # Prediksi Sentimen & Clustering
            # Kita gunakan predict_detailed untuk mendapatkan info lengkap
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
                    metadata_json={'filename': file.filename},
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
                        "source": file.filename,
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

        except Exception as e:
            db.session.rollback()
            logger.error(f"Eror Analisis: {e}")
            return jsonify({"error": str(e)}), 500

from modules.dataset_finder import DatasetFinder

# Inisialisasi DatasetFinder
dataset_finder = DatasetFinder()

@app.route('/search_and_analyze', methods=['POST'])
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
    responses:
      200:
        description: Hasil pencarian dan analisis sentimen.
    """
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Query diperlukan"}), 400
    
    query = data['query']
    logger.info(f"Menerima query pencarian: {query}")
    
    try:
        # 1. Pencarian
        logger.info("Mencari di web...")
        limit = int(data.get('limit', 100))
        raw_texts = dataset_finder.search(query, max_results=limit)
        
        model_type = data.get('model_type', 'default')
        use_hf = (model_type == 'hf')
        
        if not raw_texts:
            return jsonify({"error": "Tidak ada data ditemukan untuk topik tersebut."}), 404
            
        # 2. Preprocessing
        logger.info("Preprocessing hasil pencarian...")
        # raw_texts kini berupa list of dicts: [{'text':..., 'source':..., 'title':...}]
        texts_only = [r['text'] for r in raw_texts]
        clean_texts = preprocessor.preprocess_batch(texts_only)
        
        results = {
            "query": query,
            "total": len(raw_texts), 
            "distribution": {}, 
            "clusters": []
        }
        
        # 3. Prediksi Sentimen (menggunakan model atau lexicon)
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
                metadata_json={'query': query, 'title': title},
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
            
        return jsonify(results)

    except Exception as e:
        logger.error(f"Eror Search dan Analyze: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
