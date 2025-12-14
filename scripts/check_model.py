import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from modules.analyzer import SentimentAnalyzer

try:
    # Analyzer doesn't support model_path, it auto-loads from model_dir
    analyzer = SentimentAnalyzer(model_dir='models')
    print(f"Loaded model version: {analyzer.current_model_version}")
    if analyzer.is_trained:
        print(f"Model is trained.")
        print(f"Classes: {analyzer.classifier.classes_}")
        
        if 'nan' in analyzer.classifier.classes_:
            print("❌ BAD MODEL DETECTED: 'nan' is in classes!")
        else:
            print("✅ Model classes look clean.")
    else:
        print("Model is NOT trained.")
except Exception as e:
    print(f"Error checking model: {e}")
