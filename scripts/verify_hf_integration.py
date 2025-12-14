
import sys
import os
import logging

# Add root to path
# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_hf():
    print("Initializing Analyzer...")
    analyzer = SentimentAnalyzer()
    
    print("\nAttempting to initialize Hugging Face model (this may download ~500MB)...")
    success = analyzer.init_hf_model()
    
    if not success:
        print("FAILED to initialize HF model. Check logs/internet/dependencies.")
        return

    test_texts = [
        "Produk ini sangat bagus, saya suka sekali!",
        "Pelayanan buruk, sangat mengecewakan.",
        "Biasa saja, tidak ada yang spesial."
    ]
    
    print("\nRunning HF Prediction...")
    results = analyzer.predict_detailed_hf(test_texts)
    
    for text, res in zip(test_texts, results):
        print(f"Text: {text}")
        print(f"Result: {res['label']} (Score: {res['sentiment_score']:.4f}, Conf: {res['confidence_score']:.4f})")
        print(f"Model: {res['model_version']}")
        print("-" * 30)
        
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_hf()
