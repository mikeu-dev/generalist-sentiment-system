import pandas as pd
import numpy as np
from modules.analyzer import SentimentAnalyzer
from modules.preprocessor import TextPreprocessor

def reproduce_issue():
    print("Reproducing NaN Label Training Issue...")
    
    # 1. Simulate CSV with empty labels (like an exported search result)
    data = {
        'text': ['Ibukota bagus', 'Sangat buruk', 'Biasa saja'],
        'label': [np.nan, np.nan, np.nan] # Empty labels
    }
    df = pd.DataFrame(data)
    
    # 2. Simulate logic in /train route
    # texts = df['text'].astype(str).tolist()
    # labels = df['label'].astype(str).tolist()  <-- This is the culprit?
    
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(str).tolist()
    
    print(f"Labels converted to string: {labels}")
    
    if 'nan' in labels:
        print("✅ CONFIRMED: Labels contain string 'nan'. Model will learn this class.")
        
        # Train model
        prep = TextPreprocessor()
        clean_texts = prep.preprocess_batch(texts)
        
        analyzer = SentimentAnalyzer(model_path='model_broken.pkl')
        analyzer.train(clean_texts, labels)
        
        # Predict
        preds = analyzer.predict(clean_texts)
        print(f"Predictions: {preds}")
        
    else:
        print("❌ Cannot reproduce.")

if __name__ == "__main__":
    reproduce_issue()
