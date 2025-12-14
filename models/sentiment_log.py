from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class SentimentLog(db.Model):
    __tablename__ = 'sentiment_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    
    # Analysis Results
    label = db.Column(db.String(50))
    sentiment_score = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    cluster = db.Column(db.Integer)
    
    # Metadata
    source = db.Column(db.String(100))
    metadata_json = db.Column(db.JSON) # Named metadata_json to avoid conflict with MetaData class if any
    model_version = db.Column(db.String(50))
    
    # Tracking
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Active Learning
    is_corrected = db.Column(db.Boolean, default=False)
    original_label = db.Column(db.String(50))

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'label': self.label,
            'sentiment_score': self.sentiment_score,
            'confidence_score': self.confidence_score,
            'cluster': self.cluster,
            'source': self.source,
            'metadata': self.metadata_json,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_corrected': self.is_corrected
        }
