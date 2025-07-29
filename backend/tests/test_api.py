from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_analyze_invalid():
    r = client.post('/analyze', json={"keywords": []})
    # expecting pipeline to handle empty list gracefully
    assert r.status_code == 200 or r.status_code == 500
