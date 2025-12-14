from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import logging
from logging.config import dictConfig
from config import DevelopmentConfig, ProductionConfig
from modules.preprocessor import TextPreprocessor
from modules.analyzer import SentimentAnalyzer

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load Config
if os.environ.get('FLASK_ENV') == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize modules
logger.info("Loading modules...")
preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer()
logger.info("Modules loaded.")

@app.route('/')
def index():
    return render_template('index.html', model_trained=analyzer.is_trained)

# State Manager
from modules.training_state import TrainingStateManager

# Initialize State Manager
# Gunakan UPLOAD_FOLDER agar file persistent dan accessible
state_manager = TrainingStateManager(upload_folder=app.config['UPLOAD_FOLDER'])

def run_training_background(filepath, app_config_upload_folder):
    # Re-instantiate manager inside thread to ensure clean state handle if needed, 
    # though file locking handles concurrency.
    manager = TrainingStateManager(upload_folder=app_config_upload_folder)
    
    try:
        manager.start_training()
        
        # Read file
        manager.update_status(progress=10, message="Membaca file dataset...")
        
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Format file tidak didukung.")

        # Validate columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset harus memiliki kolom 'text' dan 'label'.")
        
        # Cleaning
        manager.update_status(progress=20, message="Membersihkan data...")
        df = df.dropna(subset=['text', 'label'])
        df = df[df['text'].astype(str).str.strip() != '']
        df = df[df['label'].astype(str).str.strip() != '']
        
        if len(df) == 0:
            raise ValueError("Dataset kosong setelah dibersihkan.")
            
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(str).tolist()
        
        # Preprocessing
        manager.update_status(progress=30, message=f"Preprocessing {len(texts)} data (bisa lama)...")
        
        clean_texts = preprocessor.preprocess_batch(texts)
        
        manager.update_status(progress=80, message="Melatih model Naive Bayes...")
        
        # Train
        analyzer.train(clean_texts, labels)
        
        result = {
            "success": True,
            "data_count": len(texts),
            "message": f"Model berhasil dilatih dengan {len(texts)} data baris."
        }
        manager.finish_training(result)
        
    except Exception as e:
        logger.error(f"Training Error: {e}")
        manager.error_training(str(e))



@app.route('/train', methods=['POST'])
def train():
    current_status = state_manager.get_status()
    if current_status["is_training"]:
        return jsonify({"error": "Training sedang berjalan. Harap tunggu."}), 409

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Start background thread
        thread = threading.Thread(target=run_training_background, args=(filepath, app.config['UPLOAD_FOLDER']))
        thread.start()
        
        return jsonify({
            "message": "Proses training dimulai di latar belakang.",
            "status": "started"
        })

@app.route('/train_status', methods=['GET'])
def get_train_status():
    return jsonify(state_manager.get_status())

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                return jsonify({"error": "Format file tidak didukung."}), 400
            
            # Flexible column name for text
            text_col = None
            for col in ['text', 'content', 'review', 'ulasan', 'komentar']:
                if col in df.columns:
                    text_col = col
                    break
            
            if not text_col:
                # If no known column, pick the first object/string column or just the first column
                text_col = df.columns[0]
            
            raw_texts = df[text_col].astype(str).tolist()
            
            logger.info("Preprocessing data for analysis...")
            clean_texts = preprocessor.preprocess_batch(raw_texts)
            
            results = {"total": len(raw_texts), "distribution": {}, "clusters": []}
            
            # Sentiment Prediction
            if analyzer.is_trained:
                predictions = analyzer.predict(clean_texts)
                df['sentiment_pred'] = predictions
                # Count distribution
                dist = df['sentiment_pred'].value_counts().to_dict()
                results['distribution'] = dist
            else:
                results['warning'] = "Model belum dilatih. Sentimen tidak diprediksi."

            # Clustering (Unsupervised)
            try:
                clusters = analyzer.cluster_topics(clean_texts, n_clusters=3)
                df['cluster'] = clusters
                
                # Analyze clusters (simple keyword extraction roughly or just counts)
                cluster_counts = df['cluster'].value_counts().to_dict()
                
                # Convert valid int64 to int for JSON serialization
                results['cluster_counts'] = {int(k): int(v) for k, v in cluster_counts.items()}
            except Exception as e:
                logger.error(f"Clustering error: {e}")
                results['cluster_error'] = str(e)

            # Prepare sample data for frontend (first 100 rows)
            preview_data = []
            for i, row in df.head(100).iterrows():
                item = {"text": row[text_col]}
                if 'sentiment_pred' in row:
                    item['sentiment'] = row['sentiment_pred']
                if 'cluster' in row:
                    item['cluster'] = int(row['cluster'])
                preview_data.append(item)
            
            results['data'] = preview_data
            
            return jsonify(results)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

from modules.dataset_finder import DatasetFinder

# Initialize DatasetFinder
dataset_finder = DatasetFinder()

@app.route('/search_and_analyze', methods=['POST'])
def search_and_analyze():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Query required"}), 400
    
    query = data['query']
    logger.info(f"Received search query: {query}")
    
    try:
        # 1. Search
        logger.info("Searching web...")
        raw_texts = dataset_finder.search(query, max_results=50)
        
        if not raw_texts:
            return jsonify({"error": "Tidak ada data ditemukan untuk topik tersebut."}), 404
            
        # 2. Preprocess
        logger.info("Preprocessing search results...")
        clean_texts = preprocessor.preprocess_batch(raw_texts)
        
        results = {
            "query": query,
            "total": len(raw_texts), 
            "distribution": {}, 
            "clusters": []
        }
        
        # 3. Predict Sentiment (using existing model or lexicon)
        # Always try to predict (analyzer now handles fallback)
        try:
            predictions = analyzer.predict(clean_texts)
            # Make a temporary DF for easy grouping
            df_temp = pd.DataFrame({'text': raw_texts, 'sentiment': predictions})
            
            dist = df_temp['sentiment'].value_counts().to_dict()
            results['distribution'] = dist
            
            if not analyzer.is_trained:
                 results['method'] = "lexicon_fallback"
                 logger.info("Using Lexicon Method (Bootstrap)")
            else:
                 results['method'] = "model_prediction"

        except Exception as e:
             results['warning'] = f"Gagal memprediksi: {str(e)}"
             logger.error(f"Prediction error: {e}")

        # 4. Clustering

        # 4. Clustering
        try:
            clusters = analyzer.cluster_topics(clean_texts, n_clusters=3)
            # Add to temp df
            # If length matches
            if len(clusters) == len(raw_texts):
                cluster_counts = pd.Series(clusters).value_counts().to_dict()
                results['cluster_counts'] = {int(k): int(v) for k, v in cluster_counts.items()}
                
                # Assign to preview
                preview_data = []
                for i, text in enumerate(raw_texts[:100]): # Limit 100
                    item = {"text": text}
                    if 'predictions' in locals():
                        item['sentiment'] = predictions[i]
                    item['cluster'] = int(clusters[i])
                    preview_data.append(item)
                results['data'] = preview_data
                
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            results['cluster_error'] = str(e)
            
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
