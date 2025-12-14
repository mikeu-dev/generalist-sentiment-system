def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Generalist Sentiment Analysis System" in response.data

def test_train_status(client):
    response = client.get('/train_status')
    assert response.status_code == 200
    json_data = response.get_json()
    assert "is_training" in json_data
    assert "progress" in json_data

def test_analyze_no_file(client):
    response = client.post('/analyze')
    assert response.status_code == 400
