from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
import pandas as pd
import pickle
import os
import datetime
import glob
import logging
from typing import List, Union, Dict, Any

from modules.lexicon_data import POSITIVE_WORDS, NEGATIVE_WORDS
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_dir: str = 'models'):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2)) # Capture bigrams
        # LinearSVC is generally better for text, CalibratedClassifierCV allows predict_proba
        svc = LinearSVC(random_state=42)
        self.classifier = CalibratedClassifierCV(svc) 
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.model_dir = model_dir
        
        # Ensure model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.is_trained: bool = False
        self.current_model_version = "v0.0.0"
        self.hf_classifier = None
        self.load_model()

    def init_hf_model(self, model_name="w11wo/indonesian-roberta-base-sentiment-classifier"):
        if not HF_AVAILABLE:
            logger.warning("Transformers library not installed. Cannot load HF model.")
            return False
            
        try:
            logger.info(f"Loading Hugging Face model: {model_name}")
            self.hf_classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
            logger.info("Hugging Face model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load HF model: {e}")
            return False

    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Melatih model dengan GridSearchCV untuk mencari hyperparameter optimal.
        """
        logger.info(f"Training model with {len(texts)} data points...")
        try:
            X = self.vectorizer.fit_transform(texts)
            
            # Define Parameter Grid for LinearSVC
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'class_weight': [None, 'balanced']
            }
            
            # Grid Search
            logger.info("Starting GridSearchCV for hyperparameter tuning...")
            grid = GridSearchCV(
                LinearSVC(random_state=42, max_iter=2000), 
                param_grid, 
                cv=5, 
                n_jobs=-1,
                scoring='accuracy'
            )
            grid.fit(X, labels)
            
            best_model = grid.best_estimator_
            logger.info(f"Best parameters found: {grid.best_params_}")
            logger.info(f"Best CV Score: {grid.best_score_:.4f}")
            
            # Wrap in CalibratedClassifierCV for probability outputs
            # We refit on the whole dataset to ensure calibration uses all data
            self.classifier = CalibratedClassifierCV(best_model, cv=5)
            self.classifier.fit(X, labels)
            
            self.is_trained = True
            self.current_model_version = f"v1.0.0_tuned_{grid.best_params_['C']}"
            self.save_model()
            logger.info("Model tuned, calibrated, trained, and saved.")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def predict(self, texts: List[str]) -> List[str]:
        """
        Legacy wrapper for predict_detailed to return only labels.
        """
        details = self.predict_detailed(texts)
        return [d['label'] for d in details]

    def predict_detailed(self, texts: List[str], use_hf: bool = False) -> List[Dict[str, Any]]:
        """
        Returns detailed prediction with confidence and scores.
        """
        logger.debug(f"predict_detailed called. is_trained={self.is_trained}, use_hf={use_hf}")
        
        if use_hf:
            return self.predict_detailed_hf(texts)
        
        if not texts:
            return []
            
        results = []

        if self.is_trained:
            # Machine Learning Prediction
            try:
                X = self.vectorizer.transform(texts)
                # Ensure classifier has predict_proba
                if hasattr(self.classifier, "predict_proba"):
                    probs = self.classifier.predict_proba(X)
                    classes = self.classifier.classes_
                    
                    # Map classes to index
                    # Assuming classes are 'positif', 'negatif', 'netral'
                    # We need to handle dynamic classes
                    
                    for i, prob_dist in enumerate(probs):
                        # Get max prob class
                        max_idx = prob_dist.argmax()
                        label = classes[max_idx]
                        confidence = float(prob_dist[max_idx])
                        
                        # Calculate Score (-1 to 1)
                        # We try to find indexes for pos/neg
                        score = 0.0
                        
                        # Safe lookup
                        pos_idx = -1
                        neg_idx = -1
                        
                        # Case insensitive lookup
                        for idx, cls_name in enumerate(classes):
                            if 'positif' in str(cls_name).lower():
                                pos_idx = idx
                            elif 'negatif' in str(cls_name).lower():
                                neg_idx = idx
                                
                        if pos_idx != -1 and neg_idx != -1:
                            score = float(prob_dist[pos_idx] - prob_dist[neg_idx])
                        elif label.lower() == 'positif':
                            score = confidence
                        elif label.lower() == 'negatif':
                            score = -confidence
                        
                        results.append({
                            "label": label,
                            "confidence_score": confidence,
                            "sentiment_score": score,
                            "model_version": self.current_model_version
                        })
                else:
                    # Fallback if no proba
                    preds = self.classifier.predict(X)
                    for p in preds:
                        results.append({
                            "label": p,
                            "confidence_score": 1.0,
                            "sentiment_score": 1.0 if 'positif' in str(p).lower() else -1.0 if 'negatif' in str(p).lower() else 0.0,
                            "model_version": self.current_model_version
                        })

            except Exception as e:
                logger.error(f"ML Prediction Error: {e}")
                # Fallback to Netral
                for _ in texts:
                    results.append({
                        "label": "netral",
                        "confidence_score": 0.0,
                        "sentiment_score": 0.0,
                        "model_version": "error_fallback"
                    })
        else:
            # Fallback: Rule-based Prediction
            logger.debug("Using Lexicon Fallback.")
            for text in texts:
                score = 0
                words = text.split()
                for word in words:
                    if word in POSITIVE_WORDS:
                        score += 1
                    elif word in NEGATIVE_WORDS:
                        score -= 1
                
                label = "netral"
                # Lexicon score logic
                norm_score = 0.0 # Normalize roughly? 
                # Let's just clamp -1 to 1
                if score > 0:
                    label = "positif"
                    norm_score = min(1.0, score * 0.2)
                elif score < 0:
                    label = "negatif"
                    norm_score = max(-1.0, score * 0.2)
                
                results.append({
                    "label": label,
                    "confidence_score": 0.5, # Rule based, unsure
                    "sentiment_score": float(norm_score),
                    "model_version": "lexicon_rule_based"
                })
            
        return results

    def predict_detailed_hf(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict utilizing Hugging Face Pipeline with robust error handling and flexible label mapping.
        """
        # Lazy load model
        if not self.hf_classifier:
             if not self.init_hf_model():
                 # Fallback to standard if init fails
                 logger.warning("HF Init failed, falling back to standard.")
                 return self.predict_detailed(texts, use_hf=False)
        
        results = []
        try:
            # Pipeline handling for batch processing
            # Truncation is important for some models (max 512 tokens usually)
            predictions = self.hf_classifier(texts, truncation=True, max_length=512)
            
            for i, pred in enumerate(predictions):
                # pred format can be {'label': '...', 'score': ...}
                label_raw = pred['label']
                confidence = pred['score']
                
                # Normalize Label to standard 'positif', 'negatif', 'netral'
                label = 'netral'
                score = 0.0
                
                # Dynamic mapping based on common HF specific outputs
                # Many indo models use 'positive', 'negative', 'neutral' or label mappings
                # We normalize to lowercase for check
                lbl_lower = label_raw.lower()

                if 'pos' in lbl_lower or 'label_0' in lbl_lower: # Label 0 often positive in some indonesian models, check specific model card if unsure. Actually w11wo/indonesian-roberta-base-sentiment-classifier: 0=positive, 1=neutral, 2=negative.
                     # Wait, let's verify w11wo model mapping:
                     # id2label: {0: "positive", 1: "neutral", 2: "negative"}
                     label = 'positif'
                     score = confidence
                elif 'neg' in lbl_lower or 'label_2' in lbl_lower:
                     label = 'negatif'
                     score = -confidence
                elif 'neu' in lbl_lower or 'label_1' in lbl_lower:
                     label = 'netral'
                     score = 0.0
                
                results.append({
                    "label": label,
                    "confidence_score": confidence,
                    "sentiment_score": score,
                    "model_version": f"hf_{self.hf_classifier.model.name_or_path}"
                })
                
        except Exception as e:
            logger.error(f"HF Prediction Error: {e}")
            # Fallback to standard model for the whole batch if critical failure
            return self.predict_detailed(texts, use_hf=False)
            
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
                 try:
                     temp_vectorizer = TfidfVectorizer()
                     X = temp_vectorizer.fit_transform(texts)
                 except ValueError:
                     logger.warning("Clustering fallback failed: Empty vocabulary, returning cluster 0.")
                     return [0] * len(texts)
        else:
             # Jika belum fit sama sekali (belum ada model), kita fit dengan data ini
             # Tapi jangan simpan ke self.vectorizer agar tidak merusak training masa depan
             # Gunakan temporary vectorizer
             try:
                 logger.debug("Clustering using temporary vectorizer (Model not trained).")
                 temp_vectorizer = TfidfVectorizer()
                 X = temp_vectorizer.fit_transform(texts)
             except ValueError:
                 logger.warning("Clustering failed: Empty vocabulary (texts might be empty or stopwords only). Returning cluster 0.")
                 return [0] * len(texts)
            
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
                self.current_model_version = os.path.basename(latest_file)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
