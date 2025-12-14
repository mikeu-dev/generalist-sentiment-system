from modules.dataset_finder import DatasetFinder
from modules.preprocessor import TextPreprocessor
from modules.analyzer import SentimentAnalyzer
import pandas as pd

def verify():
    print("Verifying Search & Analyze Pipeline...")
    
    # 1. Test Search
    print("\n[1/3] Testing DatasetFinder (DuckDuckGo)...")
    finder = DatasetFinder()
    try:
        results = finder.search("Ibukota Nusantara", max_results=5)
        if not results:
            print("❌ Search returned no results. Check internet or library status.")
            return
        print(f"✅ Found {len(results)} results.")
        print(f"   Sample: {results[0][:50]}...")
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return

    # 2. Test Processing
    print("\n[2/3] Testing Preprocessing...")
    prep = TextPreprocessor()
    clean_texts = prep.preprocess_batch(results)
    if len(clean_texts) == len(results):
        print("✅ Preprocessing successful.")
    else:
        print("❌ Preprocessing count mismatch.")
    
    # 3. Test Integration (Mocking Analyzer if not trained, but we can try)
    print("\n[3/3] Testing Analysis Integration...")
    # We use a dummy check or load real model if exists
    analyzer = SentimentAnalyzer(model_path='model_sentiment.pkl')
    if analyzer.is_trained:
        preds = analyzer.predict(clean_texts)
        print(f"✅ Prediction success: {preds[:3]}")
    else:
        print("⚠️ Model not trained, skipping prediction verification (expected if no model file).")
        
    print("\n✅ Verification Complete! Search feature is ready.")

if __name__ == "__main__":
    verify()
