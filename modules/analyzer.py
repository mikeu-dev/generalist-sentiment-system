from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
import pandas as pd
import pickle
import os
import datetime
import glob

from modules.lexicon_data import POSITIVE_WORDS, NEGATIVE_WORDS

class SentimentAnalyzer:
    def __init__(self, model_dir='models'):
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.model_dir = model_dir
        
        # Ensure model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.is_trained = False
        self.load_model()

    def train(self, texts, labels):
        """
        Melatih model Naive Bayes dengan data berlabel.
        """
        print("Training model...")
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
        self.save_model()
        print("Model trained and saved.")

    def predict(self, texts):
        """
        Memprediksi sentimen ulasan.
        Jika model belum dilatih, gunakan Rule-based (Lexicon).
        """
        print(f"DEBUG: predict called. is_trained={self.is_trained}")
        
        if self.is_trained:
            # Machine Learning Prediction
            print("DEBUG: Using ML Model.")
            # Handle empty input
            if not texts:
                return []
            
            try:
                X = self.vectorizer.transform(texts)
                res = self.classifier.predict(X)
                print(f"DEBUG: ML Result sample: {res[:5]}")
                return res
            except Exception as e:
                print(f"Prediction Error: {e}")
                # Fallback if transform fails (e.g. empty vocabulary?)
                return ["netral"] * len(texts)
        else:
            # Fallback: Rule-based Prediction
            print("DEBUG: Using Lexicon Fallback.")
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
            
            print(f"DEBUG: Lexicon Result sample: {results[:5]}")
            return results

    def cluster_topics(self, texts, n_clusters=3):
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
                 print("DEBUG: Clustering using trained vectorizer.")
                 X = self.vectorizer.transform(texts)
             except Exception as e:
                 print(f"DEBUG: Vectorizer transform failed: {e}. Fitting temporary one.")
                 temp_vectorizer = TfidfVectorizer()
                 X = temp_vectorizer.fit_transform(texts)
        else:
             # Jika belum fit sama sekali (belum ada model), kita fit dengan data ini
             # Tapi jangan simpan ke self.vectorizer agar tidak merusak training masa depan
             # Gunakan temporary vectorizer
             print("DEBUG: Clustering using temporary vectorizer (Model not trained).")
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
        return clusters

    def save_model(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_{timestamp}.pkl"
        filepath = os.path.join(self.model_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'is_trained': self.is_trained
            }, f)
        print(f"Model saved to {filepath}")

    def load_model(self):
        # Find latest model
        pattern = os.path.join(self.model_dir, "model_*.pkl")
        files = glob.glob(pattern)
        
        if not files:
            # Check for legacy model in root
            if os.path.exists('model_sentiment.pkl'):
                 print("Found legacy model_sentiment.pkl, loading...")
                 try:
                    with open('model_sentiment.pkl', 'rb') as f:
                        data = pickle.load(f)
                        self.vectorizer = data['vectorizer']
                        self.classifier = data['classifier']
                        self.is_trained = data.get('is_trained', False)
                    print("Legacy Model loaded.")
                 except Exception as e:
                    print(f"Error loading legacy model: {e}")
            else:
                 print("No model found. Starting fresh.")
            return

        # Sort by modification time
        latest_file = max(files, key=os.path.getmtime)
        print(f"Loading latest model: {latest_file}")
        
        try:
            with open(latest_file, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.classifier = data['classifier']
                self.is_trained = data.get('is_trained', False)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
