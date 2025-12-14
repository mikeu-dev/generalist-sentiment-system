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
from typing import List, Tuple, Dict, Any, Optional

from modules.lexicon_data import POSITIVE_WORDS, NEGATIVE_WORDS
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_dir: str = 'models'):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2)) # Menangkap bigram
        # LinearSVC umumnya lebih baik untuk teks, CalibratedClassifierCV memungkinkan predict_proba
        svc = LinearSVC(random_state=42)
        self.classifier = CalibratedClassifierCV(svc) 
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.model_dir = model_dir
        
        # Pastikan direktori model ada
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.is_trained: bool = False
        self.current_model_version = "v0.0.0"
        self.hf_classifier = None
        self.load_model()

    def init_hf_model(self, model_name="w11wo/indonesian-roberta-base-sentiment-classifier"):
        if not HF_AVAILABLE:
            logger.warning("Library Transformers tidak terinstal. Tidak dapat memuat model HF.")
            return False
            
        try:
            logger.info(f"Memuat model Hugging Face: {model_name}")
            self.hf_classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
            logger.info("Model Hugging Face berhasil dimuat.")
            return True
        except Exception as e:
            logger.error(f"Gagal memuat model HF: {e}")
            return False

    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Melatih model dengan GridSearchCV untuk mencari hyperparameter optimal.
        Melakukan evaluasi menggunakan train_test_split sebelum training final.
        """
        logger.info(f"Melatih model dengan {len(texts)} data points...")
        
        metrics = {}
        
        try:
            # 1. Split Data untuk Evaluasi (80% Train, 20% Test)
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
            
            # Vektorisasi Train
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)

            # Definisi Grid Parameter
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'class_weight': [None, 'balanced']
            }
            
            # 2. Grid Search pada Training Set
            logger.info("Memulai GridSearchCV untuk hyperparameter tuning...")
            grid = GridSearchCV(
                LinearSVC(random_state=42, max_iter=2000), 
                param_grid, 
                cv=5, 
                n_jobs=-1,
                scoring='accuracy'
            )
            grid.fit(X_train_vec, y_train)
            
            best_model_eval = grid.best_estimator_
            
            # 3. Evaluasi pada Test Set
            y_pred = best_model_eval.predict(X_test_vec)
            
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred, labels=grid.classes_).tolist()
            metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            metrics['classes'] = grid.classes_.tolist()
            metrics['best_params'] = grid.best_params_
            
            logger.info(f"Metrik Evaluasi: Accuracy={metrics['accuracy']:.4f}")

            # 4. Retrain pada Full Dataset untuk Production
            logger.info("Melatih ulang pada dataset penuh...")
            X_full = self.vectorizer.fit_transform(texts)
            
            # Gunakan best params dari phase sebelumnya
            final_model = LinearSVC(
                random_state=42, 
                max_iter=3000, 
                C=grid.best_params_['C'], 
                class_weight=grid.best_params_['class_weight']
            )
            
            # Bungkus dalam CalibratedClassifierCV
            self.classifier = CalibratedClassifierCV(final_model, cv=5)
            self.classifier.fit(X_full, labels)
            
            self.is_trained = True
            self.current_model_version = f"v1.1.0_tuned_{grid.best_params_['C']}"
            self.save_model()
            logger.info("Model disetel, dievaluasi, dilatih ulang, dan disimpan.")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Eror saat training: {e}")
            raise

    def predict(self, texts: List[str]) -> List[str]:
        """
        Wrapper legacy untuk predict_detailed yang hanya mengembalikan label.
        """
        details = self.predict_detailed(texts)
        return [d['label'] for d in details]

    def predict_detailed(self, texts: List[str], use_hf: bool = False) -> List[Dict[str, Any]]:
        """
        Mengembalikan prediksi detail dengan confidence dan skor sentimen.
        """
        logger.debug(f"predict_detailed dipanggil. is_trained={self.is_trained}, use_hf={use_hf}")
        
        if use_hf:
            return self.predict_detailed_hf(texts)
        
        if not texts:
            return []
            
        results = []

        if self.is_trained:
            # Prediksi Machine Learning
            try:
                X = self.vectorizer.transform(texts)
                # Pastikan classifier memiliki predict_proba
                if hasattr(self.classifier, "predict_proba"):
                    probs = self.classifier.predict_proba(X)
                    classes = self.classifier.classes_
                    
                    # Petakan kelas ke indeks
                    # Asumsi kelas adalah 'positif', 'negatif', 'netral'
                    # Kita perlu menangani kelas dinamis
                    
                    for i, prob_dist in enumerate(probs):
                        # Dapatkan kelas dengan probabilitas tertinggi
                        max_idx = prob_dist.argmax()
                        label = classes[max_idx]
                        confidence = float(prob_dist[max_idx])
                        
                        # Hitung Skor (-1 ke 1)
                        # Kita coba cari indeks untuk pos/neg
                        score = 0.0
                        
                        # Pencarian aman
                        pos_idx = -1
                        neg_idx = -1
                        
                        # Pencarian case insensitive
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
                    # Fallback jika tidak ada proba
                    preds = self.classifier.predict(X)
                    for p in preds:
                        results.append({
                            "label": p,
                            "confidence_score": 1.0,
                            "sentiment_score": 1.0 if 'positif' in str(p).lower() else -1.0 if 'negatif' in str(p).lower() else 0.0,
                            "model_version": self.current_model_version
                        })

            except Exception as e:
                logger.error(f"Eror Prediksi ML: {e}")
                # Fallback ke Netral
                for _ in texts:
                    results.append({
                        "label": "netral",
                        "confidence_score": 0.0,
                        "sentiment_score": 0.0,
                        "model_version": "error_fallback"
                    })
        else:
            # Fallback: Prediksi Berbasis Aturan (Lexicon)
            logger.debug("Menggunakan Fallback Lexicon.")
            for text in texts:
                score = 0
                words = text.split()
                for word in words:
                    if word in POSITIVE_WORDS:
                        score += 1
                    elif word in NEGATIVE_WORDS:
                        score -= 1
                
                label = "netral"
                # Logika skor Lexicon
                norm_score = 0.0 # Normalisasi kasar?
                # Mari kita batasi -1 hingga 1
                if score > 0:
                    label = "positif"
                    norm_score = min(1.0, score * 0.2)
                elif score < 0:
                    label = "negatif"
                    norm_score = max(-1.0, score * 0.2)
                
                results.append({
                    "label": label,
                    "confidence_score": 0.5, # Berbasis aturan, tidak yakin
                    "sentiment_score": float(norm_score),
                    "model_version": "lexicon_rule_based"
                })
            
        return results

    def predict_detailed_hf(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Prediksi menggunakan Hugging Face Pipeline dengan penanganan error yang kuat dan pemetaan label yang fleksibel.
        """
        # Lazy load model
        if not self.hf_classifier:
             if not self.init_hf_model():
                 # Fallback ke standar jika init gagal
                 logger.warning("Init HF gagal, fallback ke standar.")
                 return self.predict_detailed(texts, use_hf=False)
        
        results = []
        try:
            # Penanganan Pipeline untuk pemrosesan batch
            # Pemotongan (Truncation) penting untuk beberapa model (biasanya maks 512 token)
            predictions = self.hf_classifier(texts, truncation=True, max_length=512)
            
            for i, pred in enumerate(predictions):
                # format pred bisa berupa {'label': '...', 'score': ...}
                label_raw = pred['label']
                confidence = pred['score']
                
                # Normalisasi Label ke standar 'positif', 'negatif', 'netral'
                label = 'netral'
                score = 0.0
                
                # Pemetaan dinamis berdasarkan output spesifik HF umum
                # Banyak model indo menggunakan 'positive', 'negative', 'neutral' atau pemetaan label
                # Kita normalisasi ke huruf kecil untuk pemeriksaan
                lbl_lower = label_raw.lower()

                if 'pos' in lbl_lower or 'label_0' in lbl_lower: # Label 0 seringkali positif di beberapa model, cek model card jika ragu. Sebenarnya w11wo/indonesian-roberta-base-sentiment-classifier: 0=positive, 1=neutral, 2=negative.
                     # Mari verifikasi pemetaan model w11wo:
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
            logger.error(f"Eror Prediksi HF: {e}")
            # Fallback ke model standar untuk seluruh batch jika terjadi kegagalan kritis
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
                 logger.debug("Clustering menggunakan vectorizer terlatih.")
                 X = self.vectorizer.transform(texts)
             except Exception as e:
                 logger.warning(f"Transform vectorizer gagal: {e}. Fitting yang sementara.")
                 try:
                     temp_vectorizer = TfidfVectorizer()
                     X = temp_vectorizer.fit_transform(texts)
                 except ValueError:
                     logger.warning("Fallback clustering gagal: Vocabulary kosong, mengembalikan cluster 0.")
                     return [0] * len(texts)
        else:
             # Jika belum fit sama sekali (belum ada model), kita fit dengan data ini
             # Tapi jangan simpan ke self.vectorizer agar tidak merusak training masa depan
             # Gunakan temporary vectorizer
             try:
                 logger.debug("Clustering menggunakan vectorizer sementara (Model belum dilatih).")
                 temp_vectorizer = TfidfVectorizer()
                 X = temp_vectorizer.fit_transform(texts)
             except ValueError:
                 logger.warning("Clustering gagal: Vocabulary kosong (teks mungkin kosong atau stopwords saja). Mengembalikan cluster 0.")
                 return [0] * len(texts)
            
        # Pastikan cluster berbeda jika N > sampel
        n_samples = X.shape[0]
        actual_k = min(n_clusters, n_samples)
        if actual_k < 2:
             # Terlalu sedikit sampel untuk di-cluster secara bermakna, kembalikan semua 0
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
            logger.info(f"Model disimpan ke {filepath}")
        except Exception as e:
            logger.error(f"Gagal menyimpan model: {e}")

    def load_model(self) -> None:
        # Cari model terbaru
        pattern = os.path.join(self.model_dir, "model_*.pkl")
        files = glob.glob(pattern)
        
        if not files:
            # Cek model legacy di root
            if os.path.exists('model_sentiment.pkl'):
                 logger.info("Menemukan legacy model_sentiment.pkl, memuat...")
                 try:
                    with open('model_sentiment.pkl', 'rb') as f:
                        data = pickle.load(f)
                        self.vectorizer = data['vectorizer']
                        self.classifier = data['classifier']
                        self.is_trained = data.get('is_trained', False)
                    logger.info("Model Legacy dimuat.")
                 except Exception as e:
                    logger.error(f"Eror memuat model legacy: {e}")
            else:
                 logger.info("Tidak ada model ditemukan. Memulai dari awal.")
            return

        # Urutkan berdasarkan waktu modifikasi
        latest_file = max(files, key=os.path.getmtime)
        logger.info(f"Memuat model terbaru: {latest_file}")
        
        try:
            with open(latest_file, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.classifier = data['classifier']
                self.is_trained = data.get('is_trained', False)
                self.current_model_version = os.path.basename(latest_file)
            logger.info("Model berhasil dimuat.")
        except Exception as e:
            logger.error(f"Eror memuat model: {e}")
