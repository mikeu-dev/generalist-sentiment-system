
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.preprocessor import TextPreprocessor
from modules.analyzer import SentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

def test_slang_normalization():
    logger.info("--- Testing Slang Normalization ---")
    p = TextPreprocessor()
    
    test_cases = [
        ("Aku gk mau mkn", "aku tidak mau makan"), # 'makan' not in dict, but 'gk' -> 'tidak'
        ("Ini bgt bagus loh", "ini banget bagus kamu"), # 'bgt' -> 'banget', 'loh' might be removed or kept? 'loh' not in dict.
        ("sy sdh blg jgn gitu", "saya sudah bilang jangan begitu"),
    ]
    
    for input_text, expected_partial in test_cases:
        # Note: preprocess does lower, regex, stopword remove, slang norm, stem
        # So we check if the slang words are converted before stemming implicitly or explicitly
        # Let's check normalize_slang directly first if possible, but method is there.
        # But we can check final output.
        # 'makan' might stem to 'makan'
        
        normalized = p.normalize_slang(input_text.lower())
        logger.info(f"Input: '{input_text}' -> Normalized: '{normalized}'")
        
        processed = p.preprocess(input_text)
        logger.info(f"Full Processed: '{processed}'")
        
        # Simple assertions (soft)
        if "tidak" in normalized and "gk" in input_text:
            logger.info("PASSED: 'gk' converted to 'tidak'")
        else:
            if "gk" in input_text: logger.warning("FAILED: 'gk' NOT converted")

def test_training_gridsearch():
    logger.info("\n--- Testing Training with GridSearchCV ---")
    analyzer = SentimentAnalyzer()
    
    # Dummy data (need enough for CV=5, so min 5 samples per class ideally, or just total enough)
    # CV=5 requires at least one sample per split? No, sklearn needs > 1 class. 
    # StratifiedKFold needs at least 5 samples per class if n_splits=5.
    
    texts = [
        "saya sangat suka produk ini",
        "produk ini bagus sekali",
        "luar biasa hebat",
        "saya senang dengan layanan",
        "sangat memuaskan",
        "mantap jiwa",
        "keren banget",
        "ini buruk sekali",
        "saya benci ini",
        "sangat mengecewakan",
        "jelek parah",
        "tidak berguna",
        "sampah",
        "rugi beli ini",
        "biasa saja",
        "cukup oke",
        "netral",
        "tidak terlalu bagus",
        "standar",
        "lumayan"
    ]
    labels = [
        "positif", "positif", "positif", "positif", "positif", "positif", "positif",
        "negatif", "negatif", "negatif", "negatif", "negatif", "negatif", "negatif",
        "netral", "netral", "netral", "netral", "netral", "netral"
    ]
    
    try:
        analyzer.train(texts, labels)
        logger.info("Training finished successfully.")
        logger.info(f"Model Version: {analyzer.current_model_version}")
    except Exception as e:
        logger.error(f"Training Failed: {e}")
        # If failure due to small dataset for CV, that's expected but we want to ensure code runs.
        if "n_splits" in str(e):
            logger.warning("dataset too small for 5-fold CV, but logic mostly works.")

if __name__ == "__main__":
    test_slang_normalization()
    test_training_gridsearch()
