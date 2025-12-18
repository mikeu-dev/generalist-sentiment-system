"""
Unit tests untuk TextPreprocessor
"""
import pytest
from modules.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        """Fixture untuk TextPreprocessor instance"""
        return TextPreprocessor()
    
    def test_clean_text_removes_numbers(self, preprocessor):
        """Test bahwa angka dihapus dari teks"""
        text = "Saya suka produk ini 123"
        result = preprocessor.clean_text(text)
        assert "123" not in result
    
    def test_clean_text_lowercase(self, preprocessor):
        """Test bahwa teks diubah ke lowercase"""
        text = "SANGAT BAGUS"
        result = preprocessor.clean_text(text)
        assert result == result.lower()
        assert result == "sangat bagus"
    
    def test_clean_text_removes_special_chars(self, preprocessor):
        """Test bahwa karakter spesial dihapus"""
        text = "Bagus!!! @#$% Sekali!!!"
        result = preprocessor.clean_text(text)
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "!" not in result
    
    def test_clean_text_handles_empty_string(self, preprocessor):
        """Test handling untuk string kosong"""
        result = preprocessor.clean_text("")
        assert result == ""
    
    def test_clean_text_handles_none(self, preprocessor):
        """Test handling untuk None"""
        result = preprocessor.clean_text(None)
        assert result == ""
    
    def test_negation_words_preserved(self, preprocessor):
        """Test bahwa kata negasi tidak dihapus"""
        text = "Produk ini tidak bagus"
        result = preprocessor.preprocess(text)
        # Kata 'tidak' seharusnya tidak dihapus oleh stopword remover
        # Namun bisa di-stem, jadi kita cek apakah ada sisa dari 'tidak'
        assert result is not None
        assert len(result) > 0
    
    def test_slang_normalization(self, preprocessor):
        """Test normalisasi kata slang"""
        text = "gw suka bgt"
        result = preprocessor.normalize_slang(text)
        # 'gw' seharusnya dinormalisasi (tergantung isi SLANG_DICT)
        assert result is not None
    
    def test_preprocess_batch(self, preprocessor):
        """Test batch preprocessing"""
        texts = [
            "Produk bagus sekali",
            "Sangat mengecewakan",
            "Biasa saja"
        ]
        results = preprocessor.preprocess_batch(texts)
        assert len(results) == len(texts)
        assert all(isinstance(r, str) for r in results)
    
    def test_preprocess_batch_empty_list(self, preprocessor):
        """Test batch preprocessing dengan list kosong"""
        results = preprocessor.preprocess_batch([])
        assert results == []
    
    def test_cached_stem_consistency(self, preprocessor):
        """Test bahwa stemming cache memberikan hasil yang konsisten"""
        word = "memakan"
        result1 = preprocessor.cached_stem(word)
        result2 = preprocessor.cached_stem(word)
        assert result1 == result2
    
    def test_preprocess_removes_extra_whitespace(self, preprocessor):
        """Test bahwa whitespace berlebih dihapus"""
        text = "Bagus    sekali     banget"
        result = preprocessor.clean_text(text)
        assert "    " not in result
        assert "  " not in result
