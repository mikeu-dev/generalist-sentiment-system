import os
import glob
import time
import pandas as pd
from modules.analyzer import SentimentAnalyzer
import app 

def test_model_versioning():
    print("Testing Model Versioning...")
    analyzer = SentimentAnalyzer(model_dir='models')
    
    # Dummy data
    texts = ["bagus sekali", "jelek sekali", "biasa saja"]
    labels = ["positif", "negatif", "netral"]
    
    # Train
    analyzer.train(texts, labels)
    
    # Check file
    files = glob.glob("models/model_*.pkl")
    if files:
        print(f"PASS: Found model file: {files[-1]}")
    else:
        print("FAIL: No model file found in models/")
        
    # Test Clustering
    try:
        clusters = analyzer.cluster_topics(texts)
        print(f"PASS: Clustering successful. Result: {clusters}")
    except Exception as e:
        print(f"FAIL: Clustering error: {e}")

def test_async_logic():
    print("\nTesting Async Logic...")
    
    # Create temp csv
    df = pd.DataFrame({
        'text': ["test 1", "test 2"],
        'label': ["positif", "negatif"]
    })
    df.to_csv("temp_test.csv", index=False)
    
    # Reset status
    app.training_status["is_training"] = False
    
    # Run inline (not thread, just function) to test logic
    # Mock preprocessor in app to avoid full load if needed, but it should be fine
    try:
        app.run_training_background("temp_test.csv", "uploads")
        
        if app.training_status["result"]["success"]:
             print("PASS: Training finished successfully.")
        else:
             print(f"FAIL: Training failed with error: {app.training_status['result']['error']}")
             
    except Exception as e:
        print(f"FAIL: Exception during test: {e}")
        
    # Cleanup
    if os.path.exists("temp_test.csv"):
        os.remove("temp_test.csv")

if __name__ == "__main__":
    test_model_versioning()
    test_async_logic()
