#import test dependencies
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app
import io
from PIL import Image

#mock dependencies so we don't need real redis or models for api unit tests
@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch('src.api.main.Producer') as mock_producer, \
         patch('src.api.main.Predictor') as mock_predictor:

        #setup mock producer for async endpoint
        mock_prod_inst = mock_producer.return_value
        mock_prod_inst.redis = MagicMock()
        mock_prod_inst.redis.ping.return_value = True
        mock_prod_inst.push.return_value = "fake-uuid-123"

        #mock redis.get to return class prediction AND metrics
        def mock_get(key):
            if key == "result:fake-uuid-123":
                return b"3"  #simulate worker returning class 3
            elif key == "metrics:fake-uuid-123":
                #simulate metrics from worker
                import json
                return json.dumps({
                    "enqueue_time": 0.0,
                    "dequeue_time": 0.01,
                    "queue_wait_ms": 10.0,
                    "inference_time_ms": 50.0,
                    "batch_size": 1,
                    "batch_trigger": "timeout"
                })
            return None

        mock_prod_inst.redis.get = MagicMock(side_effect=mock_get)

        #setup mock predictor for sync endpoint
        mock_pred_inst = mock_predictor.return_value
        mock_pred_inst.predict_batch.return_value = [3]

        yield mock_prod_inst, mock_pred_inst

#initialize test client properly using a context manager to trigger startup events
@pytest.fixture
def client(mock_dependencies):
    with TestClient(app) as c:
        yield c

#helper function to generate dummy image upload
def create_dummy_image():
    img = Image.new('RGB', (32, 32), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

#test health endpoint
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "redis_connected": True}

#test async queued endpoint
def test_predict_async(client):
    img_bytes = create_dummy_image()
    response = client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["class_id"] == 3
    assert data["class_name"] == "cat"

#test synchronous unqueued endpoint
def test_predict_sync(client):
    img_bytes = create_dummy_image()
    response = client.post("/predict_sync", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["class_id"] == 3
    assert data["class_name"] == "cat"


#test metrics endpoint
def test_metrics_endpoint(client):
    #make a prediction to populate metrics_collector
    img_bytes = create_dummy_image()
    response = client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200

    #now query the metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert data["count"] >= 1
    assert "total_latency" in data
    assert "queue_wait" in data
    assert "inference_time" in data
    assert "avg_batch_size" in data
