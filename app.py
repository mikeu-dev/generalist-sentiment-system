from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from modules.preprocessor import TextPreprocessor
from modules.analyzer import SentimentAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize modules
print("Loading modules...")
preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer(model_path='model_sentiment.pkl')
print("Modules loaded.")

@app.route('/')
def index():
    return render_template('index.html', model_trained=analyzer.is_trained)

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # Read file (try CSV then Excel)
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                return jsonify({"error": "Format file tidak didukung. Gunakan CSV atau XLSX."}), 400

            # Validate columns
            if 'text' not in df.columns or 'label' not in df.columns:
                return jsonify({"error": "Dataset harus memiliki kolom 'text' dan 'label'."}), 400
            
            # Preprocessing
            print("Preprocessing data for training...")
            # Sample for quick response if dataset is huge, but we need full training
            texts = df['text'].astype(str).tolist()
            labels = df['label'].astype(str).tolist()
            
            clean_texts = preprocessor.preprocess_batch(texts)
            
            # Train
            analyzer.train(clean_texts, labels)
            
            return jsonify({
                "message": "Model berhasil dilatih!",
                "data_count": len(texts),
                "is_trained": True
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                return jsonify({"error": "Format file tidak didukung."}), 400
            
            # Flexible column name for text
            text_col = None
            for col in ['text', 'content', 'review', 'ulasan', 'komentar']:
                if col in df.columns:
                    text_col = col
                    break
            
            if not text_col:
                # If no known column, pick the first object/string column or just the first column
                text_col = df.columns[0]
            
            raw_texts = df[text_col].astype(str).tolist()
            
            print("Preprocessing data for analysis...")
            clean_texts = preprocessor.preprocess_batch(raw_texts)
            
            results = {"total": len(raw_texts), "distribution": {}, "clusters": []}
            
            # Sentiment Prediction
            if analyzer.is_trained:
                predictions = analyzer.predict(clean_texts)
                df['sentiment_pred'] = predictions
                # Count distribution
                dist = df['sentiment_pred'].value_counts().to_dict()
                results['distribution'] = dist
            else:
                results['warning'] = "Model belum dilatih. Sentimen tidak diprediksi."

            # Clustering (Unsupervised)
            try:
                clusters = analyzer.cluster_topics(clean_texts, n_clusters=3)
                df['cluster'] = clusters
                
                # Analyze clusters (simple keyword extraction roughly or just counts)
                cluster_counts = df['cluster'].value_counts().to_dict()
                
                # Convert valid int64 to int for JSON serialization
                results['cluster_counts'] = {int(k): int(v) for k, v in cluster_counts.items()}
            except Exception as e:
                print(f"Clustering error: {e}")
                results['cluster_error'] = str(e)

            # Prepare sample data for frontend (first 100 rows)
            preview_data = []
            for i, row in df.head(100).iterrows():
                item = {"text": row[text_col]}
                if 'sentiment_pred' in row:
                    item['sentiment'] = row['sentiment_pred']
                if 'cluster' in row:
                    item['cluster'] = int(row['cluster'])
                preview_data.append(item)
            
            results['data'] = preview_data
            
            return jsonify(results)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
