"""
Integration tests untuk API endpoints
"""
import pytest
import io
from app import app as flask_app
from models.sentiment_log import db


@pytest.fixture
def app():
    """Fixture untuk Flask app dengan testing config"""
    flask_app.config['TESTING'] = True
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    flask_app.config['WTF_CSRF_ENABLED'] = False
    
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
