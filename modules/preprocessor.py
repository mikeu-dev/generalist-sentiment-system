import re
from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import logging
from typing import List

from modules.slang_dict import SLANG_DICT

logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        logger.info("Menginisialisasi Text Preprocessor...")
        self.stemmer_factory = StemmerFactory()
        self.stemmer = self.stemmer_factory.create_stemmer()
        
        self.stopword_factory = StopWordRemoverFactory()
        
        # Kustomisasi Stopwords untuk mengecualikan kata negasi
        stopwords = self.stopword_factory.get_stop_words()
        excluded_stopwords = ['tidak', 'tak', 'bukan', 'jangan', 'kurang', 'belum', 'tidaklah']
        new_stopwords = [word for word in stopwords if word not in excluded_stopwords]
        
        # Sastrawi tidak mengizinkan penghapusan mudah, jadi kita buat kamus baru
        from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
        from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
        
        dictionary = ArrayDictionary(new_stopwords)
        self.stopword_remover = StopWordRemover(dictionary)
        logger.info("Text Preprocessor Terinisialisasi (Negasi dipertahankan).")

    def clean_text(self, text: str) -> str:
        """
        Membersihkan teks dari karakter spesial, angka, dan mengubah ke huruf kecil.
        """
        if not isinstance(text, str):
            return ""
        # Case folding (ubah ke huruf kecil)
        text = text.lower()
        # Hapus angka dan karakter non-alfabet
        text = re.sub(r'[^a-z\s]', '', text)
        # Hapus whitespace berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @lru_cache(maxsize=5000)
    def cached_stem(self, text: str) -> str:
        """
        Wrapper dengan fungsi cache untuk stemming.
        """
        return self.stemmer.stem(text)

    def preprocess(self, text: str) -> str:
        """
        Melakukan full preprocessing: cleaning -> stopword removal -> stemming.
        """
        text = self.clean_text(text)
        
        # Penghapusan Stopword
        text = self.stopword_remover.remove(text)
        
        # Normalisasi Slang (Baru)
        text = self.normalize_slang(text)
        
        # Stemming dengan cache
        text = self.cached_stem(text)
        
        return text

    def normalize_slang(self, text: str) -> str:
        """
        Mengubah kata slang/alay menjadi kata baku berdasarkan dictionary.
        """
        if not text:
            return ""
        
        words = text.split()
        normalized_words = [SLANG_DICT.get(word, word) for word in words]
        return ' '.join(normalized_words)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Memproses list teks dalam batch.
        """
        # Hapus cache jika terlalu besar untuk mencegah memory leak di long-running process
        # jika diperlukan, tapi maxsize=5000 cukup aman untuk aplikasi ini.
        return [self.preprocess(t) for t in texts]
