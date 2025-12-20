
import pandas as pd
import logging
import os
from modules.training_state import TrainingStateManager
from modules.preprocessor import TextPreprocessor
from modules.analyzer import SentimentAnalyzer

# Konfigurasi Logging (Worker biasanya butuh config sendiri atau inherit)
logger = logging.getLogger(__name__)

def run_training_background(filepath, app_config_upload_folder):
    # Instansiasi ulang manager di dalam thread/proses
    # Oper URL Redis jika perlu, tapi default localhost sudah cukup
    manager = TrainingStateManager(upload_folder=app_config_upload_folder)
    
    # Inisialisasi modul baru
    preprocessor = TextPreprocessor()
    analyzer = SentimentAnalyzer()
    
    try:
        manager.start_training()
        
        # Baca file
        manager.update_status(progress=10, message="Membaca file dataset...")
        
        if filepath.endswith('.csv'):
            try:
                df = pd.read_csv(filepath)
            except Exception:
                manager.update_status(progress=15, message="Menggunakan mode baca robust (format CSV kompleks)...")
                df = pd.read_csv(filepath, engine='python', on_bad_lines='skip', quotechar='"', encoding_errors='ignore')
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Format file tidak didukung.")

        # Validasi kolom
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset harus memiliki kolom 'text' dan 'label'.")
        
        # Pembersihan Data
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
        
        # Latih Model
        metrics = analyzer.train(clean_texts, labels)
        
        result = {
            "success": True,
            "data_count": len(texts),
            "message": f"Model berhasil dilatih dengan {len(texts)} data baris.",
            "metrics": metrics
        }
        manager.finish_training(result)
        
    except Exception as e:
        logger.error(f"Eror Training: {e}")
        manager.error_training(str(e))

def update_topic_sentiment(topic_id, app_config_db_uri="sqlite:///instance/sentiment.db"):
    """
    Background job to update a specific monitored topic.
    Need database context.
    """
    from flask import Flask
    from models.sentiment_log import db
    from models.topic_models import MonitoredTopic, TopicSnapshot
    from modules.dataset_finder import DatasetFinder
    from modules.preprocessor import TextPreprocessor
    from modules.analyzer import SentimentAnalyzer
    from datetime import datetime
    
    # Create minimal app context
    app = Flask(__name__)
    # Ensure URI is absolute or correct relative path if running from worker
    # Worker runs from root usually.
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI', app_config_db_uri)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        try:
            logger.info(f"Updating topic ID: {topic_id}")
            topic = MonitoredTopic.query.get(topic_id)
            if not topic:
                logger.error("Topic not found")
                return
            
            # Init Modules
            finder = DatasetFinder()
            preprocessor = TextPreprocessor()
            analyzer = SentimentAnalyzer()
            
            # Fetch Data
            # Note: In a real monitoring system, we would query by 'since' date.
            # Here we just fetch fresh results to get current sentiment snapshot.
            query = topic.search_query
            logger.info(f"Searching for: {query}")
            results = finder.search(query, max_results=100) # Limit for speed
            
            if not results:
                logger.warning(f"No results found for {query}")
                return
            
            # Analyze
            raw_texts = [r['text'] for r in results]
            clean_texts = preprocessor.preprocess_batch(raw_texts)
            details = analyzer.predict_detailed(clean_texts)
            
            # Calculate Stats
            pos = 0
            neg = 0
            neu = 0
            total_score = 0
            
            for d in details:
                if d['label'] == 'positif': pos += 1
                elif d['label'] == 'negatif': neg += 1
                else: neu += 1
                total_score += d['sentiment_score']
            
            avg_score = total_score / len(details) if details else 0
            
            # Save Snapshot
            snapshot = TopicSnapshot(
                topic_id=topic.id,
                positive_count=pos,
                negative_count=neg,
                neutral_count=neu,
                total_samples=len(details),
                sentiment_score_avg=avg_score,
                timestamp=datetime.utcnow()
            )
            
            # Update Topic Last Updated
            topic.last_updated = datetime.utcnow()
            
            db.session.add(snapshot)
            db.session.commit()
            logger.info(f"Snapshot saved for topic {topic.name}. Score: {avg_score}")
            
        except Exception as e:
            logger.error(f"Error updating topic {topic_id}: {e}", exc_info=True)
            db.session.rollback()
