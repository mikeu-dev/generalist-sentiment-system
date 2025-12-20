from models.sentiment_log import db
from datetime import datetime
import json

class MonitoredTopic(db.Model):
    __tablename__ = 'monitored_topics'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    search_query = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    snapshots = db.relationship('TopicSnapshot', backref='topic', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'query': self.search_query,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'is_active': self.is_active
        }

class TopicSnapshot(db.Model):
    __tablename__ = 'topic_snapshots'
    
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('monitored_topics.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Store aggregated stats
    positive_count = db.Column(db.Integer, default=0)
    negative_count = db.Column(db.Integer, default=0)
    neutral_count = db.Column(db.Integer, default=0)
    total_samples = db.Column(db.Integer, default=0)
    sentiment_score_avg = db.Column(db.Float, default=0.0)
    
    # Optional: Store raw JSON distribution if we want more detail later
    # sentiment_distribution = db.Column(db.String) # JSON String
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'positive': self.positive_count,
            'negative': self.negative_count,
            'neutral': self.neutral_count,
            'total': self.total_samples,
            'score_avg': self.sentiment_score_avg
        }
