import re
from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import logging
from typing import List

logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        logger.info("Initializing Text Preprocessor...")
        self.stemmer_factory = StemmerFactory()
        self.stemmer = self.stemmer_factory.create_stemmer()
        
        self.stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = self.stopword_factory.create_stop_word_remover()
        logger.info("Text Preprocessor Initialized.")

    def clean_text(self, text: str) -> str:
        """
        Membersihkan teks dari karakter spesial, angka, dan mengubah ke huruf kecil.
        """
        if not isinstance(text, str):
            return ""
        # Case folding
        text = text.lower()
        # Hapus angka dan karakter non-alfabet
        text = re.sub(r'[^a-z\s]', '', text)
        # Hapus whitespace berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @lru_cache(maxsize=5000)
    def cached_stem(self, text: str) -> str:
        """
        Wrapper cached untuk stemming.
        """
        return self.stemmer.stem(text)

    def preprocess(self, text: str) -> str:
        """
        Melakukan full preprocessing: cleaning -> stopword removal -> stemming.
        """
        text = self.clean_text(text)
        
        # Stopword removal
        text = self.stopword_remover.remove(text)
        
        # Stemming with cache
        text = self.cached_stem(text)
        
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Memproses list teks.
        """
        # Hapus cache jika terlalu besar untuk mencegah memory leak di long-running process
        # jika diperlukan, tapi maxsize=5000 cukup aman untuk aplikasi ini.
        return [self.preprocess(t) for t in texts]
