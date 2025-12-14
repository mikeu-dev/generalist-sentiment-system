from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
import pandas as pd
import pickle
import os
import datetime
import glob
import logging
from typing import List, Union, Dict, Any

from modules.lexicon_data import POSITIVE_WORDS, NEGATIVE_WORDS

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_dir: str = 'models'):
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.model_dir = model_dir
        
        # Ensure model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.is_trained: bool = False
        self.load_model()

    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Melatih model Naive Bayes dengan data berlabel.
        """
        logger.info("Training model...")
        try:
            X = self.vectorizer.fit_transform(texts)
            self.classifier.fit(X, labels)
            self.is_trained = True
            self.save_model()
            logger.info("Model trained and saved.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def predict(self, texts: List[str]) -> List[str]:
        """
        Memprediksi sentimen ulasan.
        Jika model belum dilatih, gunakan Rule-based (Lexicon).
        """
        logger.debug(f"predict called. is_trained={self.is_trained}")
        
        if self.is_trained:
            # Machine Learning Prediction
            logger.debug("Using ML Model.")
            # Handle empty input
            if not texts:
                return []
            
            try:
                X = self.vectorizer.transform(texts)
                res = self.classifier.predict(X)
                logger.debug(f"ML Result sample: {res[:5]}")
                return list(res)
            except Exception as e:
                logger.error(f"Prediction Error: {e}")
                # Fallback if transform fails (e.g. empty vocabulary?)
                return ["netral"] * len(texts)
        else:
            # Fallback: Rule-based Prediction
            logger.debug("Using Lexicon Fallback.")
            results = []
            for text in texts:
                score = 0
                words = text.split()
                for word in words:
                    if word in POSITIVE_WORDS:
                        score += 1
                    elif word in NEGATIVE_WORDS:
                        score -= 1
                
                if score > 0:
                    results.append("positif")
                elif score < 0:
                    results.append("negatif")
                else:
                    results.append("netral")
            
            logger.debug(f"Lexicon Result sample: {results[:5]}")
            return results

    def cluster_topics(self, texts: List[str], n_clusters: int = 3) -> List[int]:
        """
        Mengelompokkan teks ke dalam topik menggunakan K-Means.
        """
        if not texts:
            return []

        # Gunakan vectorizer yang sudah ada jika vocabulary sudah terbentuk
        # Ini penting agar fitur konsisten dengan data training (jika ada)
        if hasattr(self.vectorizer, 'vocabulary_') and self.is_trained:
             # Transform menggunakan vocabulary yang sudah ada
             try:
                 logger.debug("Clustering using trained vectorizer.")
                 X = self.vectorizer.transform(texts)
             except Exception as e:
                 logger.warning(f"Vectorizer transform failed: {e}. Fitting temporary one.")
                 temp_vectorizer = TfidfVectorizer()
                 X = temp_vectorizer.fit_transform(texts)
        else:
             # Jika belum fit sama sekali (belum ada model), kita fit dengan data ini
             # Tapi jangan simpan ke self.vectorizer agar tidak merusak training masa depan
             # Gunakan temporary vectorizer
             logger.debug("Clustering using temporary vectorizer (Model not trained).")
             temp_vectorizer = TfidfVectorizer()
             X = temp_vectorizer.fit_transform(texts)
            
        # Ensure distinct clusters if N > samples
        n_samples = X.shape[0]
        actual_k = min(n_clusters, n_samples)
        if actual_k < 2:
             # Too few samples to cluster meaningfully, return all 0
             return [0] * n_samples

        kmeans_local = KMeans(n_clusters=actual_k, random_state=42)
        clusters = kmeans_local.fit_predict(X)
        return list(clusters)

    def save_model(self) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_{timestamp}.pkl"
        filepath = os.path.join(self.model_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'classifier': self.classifier,
                    'is_trained': self.is_trained
                }, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self) -> None:
        # Find latest model
        pattern = os.path.join(self.model_dir, "model_*.pkl")
        files = glob.glob(pattern)
        
        if not files:
            # Check for legacy model in root
            if os.path.exists('model_sentiment.pkl'):
                 logger.info("Found legacy model_sentiment.pkl, loading...")
                 try:
                    with open('model_sentiment.pkl', 'rb') as f:
                        data = pickle.load(f)
                        self.vectorizer = data['vectorizer']
                        self.classifier = data['classifier']
                        self.is_trained = data.get('is_trained', False)
                    logger.info("Legacy Model loaded.")
                 except Exception as e:
                    logger.error(f"Error loading legacy model: {e}")
            else:
                 logger.info("No model found. Starting fresh.")
            return

        # Sort by modification time
        latest_file = max(files, key=os.path.getmtime)
        logger.info(f"Loading latest model: {latest_file}")
        
        try:
            with open(latest_file, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.classifier = data['classifier']
                self.is_trained = data.get('is_trained', False)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
