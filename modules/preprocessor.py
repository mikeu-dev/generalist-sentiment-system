import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TextPreprocessor:
    def __init__(self):
        print("Initializing Text Preprocessor...")
        self.stemmer_factory = StemmerFactory()
        self.stemmer = self.stemmer_factory.create_stemmer()
        
        self.stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = self.stopword_factory.create_stop_word_remover()
        print("Text Preprocessor Initialized.")

    def clean_text(self, text):
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

    def preprocess(self, text):
        """
        Melakukan full preprocessing: cleaning -> stopword removal -> stemming.
        """
        text = self.clean_text(text)
        
        # Stopword removal
        text = self.stopword_remover.remove(text)
        
        # Stemming
        # Note: Stemming is slow. In production, consider caching or lighter stemming.
        text = self.stemmer.stem(text)
        
        return text

    def preprocess_batch(self, texts):
        """
        Memproses list teks.
        """
        return [self.preprocess(t) for t in texts]
