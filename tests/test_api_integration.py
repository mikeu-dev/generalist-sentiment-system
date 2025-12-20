"""
Integration tests untuk API endpoints
"""
import pytest
import io
from app import app as flask_app
from models.sentiment_log import db
# Import all models to ensure db.create_all() works
import models.topic_models


@pytest.fixture
def app():
    """Fixture untuk Flask app dengan testing config"""
    flask_app.config['TESTING'] = True
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    flask_app.config['RATELIMIT_ENABLED'] = False
    
    # Disable limiter explicitly
    from app import limiter, queue
    limiter.enabled = False
    
    # Mock queue
    from unittest.mock import MagicMock
    mock_job = MagicMock()
    mock_job.id = "mock-job-id"
    queue.enqueue = MagicMock(return_value=mock_job)
    
    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """Fixture untuk test client"""
    return app.test_client()


class TestHealthEndpoint:
    def test_health_check_success(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code in [200, 503]
        data = response.get_json()
        assert 'status' in data
        assert 'checks' in data
        assert 'timestamp' in data


class TestMetricsEndpoint:
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get('/metrics')
        assert response.status_code == 200
        data = response.get_json()
        assert 'total_analyses' in data
        assert 'model_version' in data


class TestAnalyzeEndpoint:
    def test_analyze_without_file(self, client):
        """Test analyze endpoint tanpa file"""
        response = client.post('/analyze')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_analyze_with_empty_filename(self, client):
        """Test analyze dengan filename kosong"""
        data = {
            'file': (io.BytesIO(b''), ''),
        }
        response = client.post('/analyze', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
    
    def test_analyze_with_invalid_extension(self, client):
        """Test analyze dengan extension tidak valid"""
        data = {
            'file': (io.BytesIO(b'malicious content'), 'test.exe'),
        }
        response = client.post('/analyze', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
        response_data = response.get_json()
        assert 'error' in response_data


class TestSearchEndpoint:
    def test_search_without_query(self, client):
        """Test search endpoint tanpa query"""
        response = client.post('/search_and_analyze',
                              json={},
                              content_type='application/json')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_search_with_empty_query(self, client):
        """Test search dengan query kosong"""
        response = client.post('/search_and_analyze',
                              json={'query': ''},
                              content_type='application/json')
        assert response.status_code == 400
    
    def test_search_with_too_short_query(self, client):
        """Test search dengan query terlalu pendek"""
        response = client.post('/search_and_analyze',
                              json={'query': 'a'},
                              content_type='application/json')
        assert response.status_code == 400


class TestTrainEndpoint:
    def test_train_without_file(self, client):
        """Test train endpoint tanpa file"""
        response = client.post('/train')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_train_with_invalid_extension(self, client):
        """Test train dengan extension tidak valid"""
        data = {
            'file': (io.BytesIO(b'content'), 'model.pkl'),
        }
        response = client.post('/train', data=data, content_type='multipart/form-data')
        assert response.status_code == 400


class TestTrainStatusEndpoint:
    def test_train_status(self, client):
        """Test train status endpoint"""
        response = client.get('/train_status')
        assert response.status_code == 200
        data = response.get_json()
        assert 'is_training' in data

class TestExportEndpoint:
    def test_export_invalid_batch_id(self, client):
        """Test export dengan batch_id yang tidak ada"""
        response = client.get('/api/export/nonexistent_id')
        assert response.status_code == 404
    
    def test_export_excel(self, client, app):
        """Test export ke Excel"""
        # Seed data
        import uuid
        from models.sentiment_log import SentimentLog, db
        import json
        
        batch_id = str(uuid.uuid4())
        log = SentimentLog(
            text="Test export data",
            label="positif",
            sentiment_score=0.9,
            confidence_score=0.95,
            cluster=0,
            source="test",
            metadata_json=json.dumps({'batch_id': batch_id}),
            model_version="test_v1"
        )
        with app.app_context():
            db.session.add(log)
            db.session.commit()
            
            # Test Excel
            response = client.get(f'/api/export/{batch_id}?format=excel')
            assert response.status_code == 200
            assert response.headers['Content-Type'] == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            assert f"attachment; filename=sentiment_data_{batch_id[:8]}.xlsx" in response.headers['Content-Disposition']

    def test_export_pdf(self, client, app):
        """Test export ke PDF"""
        # Seed data
        import uuid
        from models.sentiment_log import SentimentLog, db
        import json
        
        batch_id = str(uuid.uuid4())
        log = SentimentLog(
            text="Test export data pdf",
            label="positif",
            sentiment_score=0.9,
            confidence_score=0.95,
            cluster=0,
            source="test",
            metadata_json=json.dumps({'batch_id': batch_id}),
            model_version="test_v1"
        )
        with app.app_context():
            db.session.add(log)
            db.session.commit()
            
            # Test PDF
            response = client.get(f'/api/export/{batch_id}?format=pdf')
            assert response.status_code == 200
            assert response.headers['Content-Type'] == 'application/pdf'
            assert f"attachment; filename=sentiment_report_{batch_id[:8]}.pdf" in response.headers['Content-Disposition']

class TestMonitoringEndpoint:
    def test_create_topic(self, client):
        """Test membuat topik monitoring baru"""
        response = client.post('/api/monitoring/topics', json={
            'name': 'Test Topic',
            'query': 'test query'
        })
        assert response.status_code == 201
        data = response.get_json()
        assert data['name'] == 'Test Topic'
        assert data['query'] == 'test query'
        assert 'id' in data

    def test_list_topics(self, client):
        """Test list topik"""
        # Create one first
        client.post('/api/monitoring/topics', json={'name': 'T1', 'query': 'Q1'})
        
        response = client.get('/api/monitoring/topics')
        if response.status_code != 200:
            print(f"DEBUG ERROR: {response.status_code}, {response.data}")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_delete_topic(self, client):
        """Test hapus topik"""
        # Create
        res = client.post('/api/monitoring/topics', json={'name': 'To Delete', 'query': 'QDel'})
        topic_id = res.get_json()['id']
        
        # Delete
        del_res = client.delete(f'/api/monitoring/topics/{topic_id}')
        assert del_res.status_code == 200
        
        # Verify gone
        get_res = client.get('/api/monitoring/topics')
        topics = get_res.get_json()
        assert not any(t['id'] == topic_id for t in topics)

    def test_refresh_topic(self, client):
        """Test refresh trigger"""
        # Create
        res = client.post('/api/monitoring/topics', json={'name': 'To Refresh', 'query': 'QRef'})
        topic_id = res.get_json()['id']
        
        # Refresh
        ref_res = client.post(f'/api/monitoring/topics/{topic_id}/refresh')
        assert ref_res.status_code == 200
        data = ref_res.get_json()
        assert 'job_id' in data
