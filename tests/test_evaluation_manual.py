
import sys
import os
import logging
from typing import List

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.analyzer import SentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test")

def test_training_metrics():
    print("Testing Training Metrics...")
    
    # Create manual dataset
    texts = [
        "Produk ini sangat bagus dan memuaskan", "Saya sangat suka dengan pelayanan ini",
        "Kualitas barang oke banget", "Recommended banget!", "Suka sekali",
        "Jelek sekali produknya", "Sangat mengecewakan", "Tidak recommended",
        "Pelayanan lambat", "Barang rusak parah",
        "Biasa saja sih", "Standar aja", "Lumayan lah", "Cukup oke", "Not bad",
        "Sangat bagus", "Luar biasa", "Parah", "Buruk", "Kecewa"
    ]
    labels = [
        "positif", "positif", "positif", "positif", "positif",
        "negatif", "negatif", "negatif", "negatif", "negatif",
        "netral", "netral", "netral", "netral", "netral",
        "positif", "positif", "negatif", "negatif", "negatif"
    ]
    
    analyzer = SentimentAnalyzer(model_dir='tests/test_models')
    
    try:
        metrics = analyzer.train(texts, labels)
        print("\n--- Training Result Metrics ---")
        print(f"Accuracy: {metrics.get('accuracy')}")
        print(f"Confusion Matrix: {metrics.get('confusion_matrix')}")
        print(f"Classification Report: {metrics.get('classification_report')}")
        print(f"Classes: {metrics.get('classes')}")
        
        if 'confusion_matrix' in metrics and 'accuracy' in metrics:
            print("\nSUCCESS: Metrics returned correctly.")
        else:
            print("\nFAILURE: Metrics missing.")
            
    except Exception as e:
        print(f"\nERROR: Training failed with {e}")
        raise

if __name__ == "__main__":
    test_training_metrics()
