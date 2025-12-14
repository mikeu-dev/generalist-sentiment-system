
import pandas as pd
import logging
import os
from modules.training_state import TrainingStateManager
from modules.preprocessor import TextPreprocessor
from modules.analyzer import SentimentAnalyzer

# Configure Logging (Worker needs its own config usually, or inherits)
logger = logging.getLogger(__name__)

def run_training_background(filepath, app_config_upload_folder):
    # Re-instantiate manager inside thread/process
    # Pass Redis URL if needed, but defaults to localhost which is fine for now
    manager = TrainingStateManager(upload_folder=app_config_upload_folder)
    
    # Initialize modules fresh
    preprocessor = TextPreprocessor()
    analyzer = SentimentAnalyzer()
    
    try:
        manager.start_training()
        
        # Read file
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
