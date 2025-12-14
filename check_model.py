import pickle
from modules.analyzer import SentimentAnalyzer

try:
    analyzer = SentimentAnalyzer(model_path='model_sentiment.pkl')
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
