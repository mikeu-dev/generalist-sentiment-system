"""
Health check dan monitoring endpoints
"""
from flask import jsonify
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def register_monitoring_routes(app, db, redis_conn, analyzer, SentimentLog):
    """Register health check dan metrics endpoints"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint untuk monitoring sistem."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        # Check database
        try:
            db.session.execute('SELECT 1')
            health_status['checks']['database'] = 'ok'
        except Exception as e:
            health_status['checks']['database'] = 'error'
            health_status['status'] = 'unhealthy'
            logger.error(f"Database health check failed: {e}")
        
        # Check Redis
        try:
            redis_conn.ping()
            health_status['checks']['redis'] = 'ok'
        except Exception as e:
            health_status['checks']['redis'] = 'error'
            health_status['status'] = 'unhealthy'
            logger.error(f"Redis health check failed: {e}")
        
        # Check model
        health_status['checks']['model'] = 'trained' if analyzer.is_trained else 'not_trained'
        health_status['checks']['model_version'] = analyzer.current_model_version
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code

    @app.route('/metrics', methods=['GET'])
    def metrics():
        """Metrics endpoint untuk monitoring."""
        try:
            total_analyses = SentimentLog.query.count()
            sentiment_dist = db.session.query(
                SentimentLog.label,
                db.func.count(SentimentLog.id)
            ).group_by(SentimentLog.label).all()
            
            return jsonify({
                'total_analyses': total_analyses,
                'model_version': analyzer.current_model_version,
                'model_trained': analyzer.is_trained,
                'sentiment_distribution': {label: count for label, count in sentiment_dist}
            })
        except Exception as e:
            logger.error(f"Error in metrics endpoint: {e}")
            return jsonify({"error": "Gagal mengambil metrics"}), 500
