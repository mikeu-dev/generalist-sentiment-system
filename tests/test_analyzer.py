"""
Unit tests untuk SentimentAnalyzer
"""
import pytest
import os
import tempfile
from modules.analyzer import SentimentAnalyzer


class TestSentimentAnalyzer:
    @pytest.fixture
    def analyzer(self):
        """Fixture untuk SentimentAnalyzer instance dengan temp directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = SentimentAnalyzer(model_dir=tmpdir)
            yield analyzer
    
    def test_analyzer_initialization(self, analyzer):
        """Test bahwa analyzer terinisialisasi dengan benar"""
        assert analyzer is not None
        assert hasattr(analyzer, 'vectorizer')
        assert hasattr(analyzer, 'classifier')
        assert hasattr(analyzer, 'kmeans')
    
    def test_predict_detailed_without_training(self, analyzer):
        """Test prediksi tanpa training (fallback ke lexicon)"""
        texts = ["bagus sekali", "jelek banget", "biasa saja"]
        results = analyzer.predict_detailed(texts)
        
        assert len(results) == len(texts)
        assert all('label' in r for r in results)
        assert all('confidence_score' in r for r in results)
        assert all('sentiment_score' in r for r in results)
        assert all('model_version' in r for r in results)
    
    def test_predict_detailed_labels(self, analyzer):
        """Test bahwa label yang dikembalikan valid"""
        texts = ["sangat bagus", "sangat buruk"]
        results = analyzer.predict_detailed(texts)
        
        valid_labels = ['positif', 'negatif', 'netral']
        assert all(r['label'] in valid_labels for r in results)
    
    def test_predict_detailed_empty_list(self, analyzer):
        """Test prediksi dengan list kosong"""
        results = analyzer.predict_detailed([])
        assert results == []
    
    def test_cluster_topics_basic(self, analyzer):
        """Test clustering topik"""
        texts = [
            "produk bagus harga murah",
            "produk jelek harga mahal",
            "kualitas bagus pelayanan cepat"
        ]
        clusters = analyzer.cluster_topics(texts, n_clusters=2)
        
        assert len(clusters) == len(texts)
        assert all(isinstance(c, int) for c in clusters)
    
    def test_cluster_topics_empty_list(self, analyzer):
        """Test clustering dengan list kosong"""
        clusters = analyzer.cluster_topics([])
        assert clusters == []
    
    def test_cluster_topics_single_item(self, analyzer):
        """Test clustering dengan satu item"""
        texts = ["produk bagus"]
        clusters = analyzer.cluster_topics(texts, n_clusters=3)
        assert len(clusters) == 1
        assert clusters[0] == 0
    
    def test_train_basic(self, analyzer):
        """Test training dengan dataset kecil"""
        texts = [
            "bagus sekali saya suka",
            "jelek sekali saya kecewa",
            "biasa saja tidak istimewa",
            "sangat memuaskan",
            "sangat mengecewakan",
            "cukup baik"
        ]
        labels = ["positif", "negatif", "netral", "positif", "negatif", "netral"]
        
        metrics = analyzer.train(texts, labels)
        
        assert analyzer.is_trained
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert 'classification_report' in metrics
        assert metrics['accuracy'] > 0
    
    def test_predict_after_training(self, analyzer):
        """Test prediksi setelah training"""
        # Training
        texts = [
            "bagus sekali",
            "jelek sekali",
            "biasa saja",
            "sangat baik",
            "sangat buruk",
            "cukup"
        ]
        labels = ["positif", "negatif", "netral", "positif", "negatif", "netral"]
        analyzer.train(texts, labels)
        
        # Prediksi
        test_texts = ["bagus", "buruk"]
        results = analyzer.predict_detailed(test_texts)
        
        assert len(results) == len(test_texts)
        assert all('model_version' in r for r in results)
        assert all(r['model_version'] != 'lexicon_rule_based' for r in results)
    
    def test_save_and_load_model(self, analyzer):
        """Test save dan load model"""
        # Training
        texts = ["bagus", "jelek", "biasa", "baik", "buruk", "ok"]
        labels = ["positif", "negatif", "netral", "positif", "negatif", "netral"]
        analyzer.train(texts, labels)
        
        # Save
        analyzer.save_model()
        
        # Load dengan analyzer baru
        new_analyzer = SentimentAnalyzer(model_dir=analyzer.model_dir)
        
        assert new_analyzer.is_trained
    
    def test_confidence_score_range(self, analyzer):
        """Test bahwa confidence score dalam range 0-1"""
        texts = ["bagus sekali", "jelek banget"]
        results = analyzer.predict_detailed(texts)
        
        for result in results:
            assert 0 <= result['confidence_score'] <= 1
    
    def test_sentiment_score_range(self, analyzer):
        """Test bahwa sentiment score dalam range -1 to 1"""
        texts = ["bagus sekali", "jelek banget", "biasa"]
        results = analyzer.predict_detailed(texts)
        
        for result in results:
            assert -1 <= result['sentiment_score'] <= 1
