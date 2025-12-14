from modules.preprocessor import TextPreprocessor
from modules.analyzer import SentimentAnalyzer
import pytest

def test_preprocessor_clean():
    p = TextPreprocessor()
    raw = "Ini adalah TEST!! 123"
    clean = p.clean_text(raw)
    assert clean == "ini adalah test"

def test_preprocessor_stem():
    p = TextPreprocessor()
    # Sastrawi stemming example
    stemmed = p.cached_stem("memakan")
    assert stemmed == "makan"

def test_analyzer_predict_untrained():
    a = SentimentAnalyzer()
    a.is_trained = False
    
    # Validation for lexicon fallback
    # "bagus" is usually positive, "jelek" is negative
    res = a.predict(["barang ini bagus", "barang ini jelek", "biasa saja"])
    assert len(res) == 3
    assert res[0] == "positif"
    assert res[1] == "negatif"
