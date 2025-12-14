import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from modules.preprocessor import TextPreprocessor
from modules.analyzer import SentimentAnalyzer
import os

def test_system():
    print("Testing System Logic...")
    
    try:
        # 1. Initialize
        print("[1/7] Initializing Modules...")
        prep = TextPreprocessor()
        # Use a temp dir for test model
        test_model_dir = 'test_models_temp'
        if not os.path.exists(test_model_dir):
            os.makedirs(test_model_dir)
            
        analyzer = SentimentAnalyzer(model_dir=test_model_dir)
        
        # 2. Load Training Data
        print("[2/7] Loading Training Data...")
        if not os.path.exists('data/samples/sample_training.csv'):
            print("❌ Error: data/samples/sample_training.csv not found.")
            return
        df_train = pd.read_csv('data/samples/sample_training.csv')
        texts = df_train['text'].astype(str).tolist()
        labels = df_train['label'].astype(str).tolist()
        
        # 3. Preprocess
        print("[3/7] Preprocessing Training Data...")
        clean_texts = prep.preprocess_batch(texts)
        print(f"   Sample clean: '{clean_texts[0]}'")
        
        # 4. Train
        print("[4/7] Training Model...")
        analyzer.train(clean_texts, labels)
        
        # 5. Load Review Data
        print("[5/7] Loading Review Data...")
        if not os.path.exists('data/samples/sample_reviews.csv'):
            print("❌ Error: data/samples/sample_reviews.csv not found.")
            return
        df_test = pd.read_csv('data/samples/sample_reviews.csv')
        test_texts = df_test['text'].astype(str).tolist()
        
        # 6. Predict
        print("[6/7] Predicting Sentiment...")
        clean_test_texts = prep.preprocess_batch(test_texts)
        preds = analyzer.predict(clean_test_texts)
        print(f"   Predictions: {preds[:5]}...")
        
        # 7. Cluster
        print("[7/7] Clustering Topik...")
        clusters = analyzer.cluster_topics(clean_test_texts, n_clusters=2)
        print(f"   Clusters: {clusters[:5]}...")
        
        print("\n✅ Verification Successful! Logika backend berjalan normal.")
        
        # Cleanup
        if os.path.exists('test_models_temp'):
            import shutil
            shutil.rmtree('test_models_temp')
            
    except Exception as e:
        print(f"\n❌ Verification Failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()
