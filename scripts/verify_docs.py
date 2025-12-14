
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, train, analyze, search_and_analyze
from modules.analyzer import SentimentAnalyzer
from modules.preprocessor import TextPreprocessor

def check_docstring(obj, name):
    doc = obj.__doc__
    if not doc:
        print(f"FAILED: {name} has no docstring.")
        return False
    
    # Simple check for Indonesian words
    keywords = ['melatih', 'analisis', 'sentimen', 'data', 'mengembalikan', 'membersihkan']
    found = any(k in doc.lower() for k in keywords)
    
    if found:
        print(f"PASSED: {name} docstring seems to be in Indonesian.")
        return True
    else:
        print(f"WARNING: {name} docstring might not be in Indonesian. Content: {doc[:50]}...")
        return False

def main():
    print("Verifying Docstrings...")
    all_passed = True
    
    # Check App Routes
    if not check_docstring(train, "app.train"): all_passed = False
    if not check_docstring(analyze, "app.analyze"): all_passed = False
    if not check_docstring(search_and_analyze, "app.search_and_analyze"): all_passed = False
    
    # Check Modules
    analyzer = SentimentAnalyzer()
    preprocessor = TextPreprocessor()
    
    if not check_docstring(analyzer.train, "SentimentAnalyzer.train"): all_passed = False
    if not check_docstring(analyzer.predict_detailed, "SentimentAnalyzer.predict_detailed"): all_passed = False
    if not check_docstring(preprocessor.clean_text, "TextPreprocessor.clean_text"): all_passed = False
    
    if all_passed:
        print("\nSUCCESS: All critical docstrings verified.")
    else:
        print("\nFAILURE: Some docstrings failed verification.")
        sys.exit(1)

if __name__ == "__main__":
    main()
