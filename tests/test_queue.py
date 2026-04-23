#tests/test_queue.py
import pytest
import torch
import json
import fakeredis
from unittest.mock import patch
from src.serving.producer import Producer
from src.serving.consumer import Consumer

@pytest.fixture
def fake_redis():
    #create a fake redis instance for testing
    return fakeredis.FakeRedis(decode_responses=True)

@patch('src.serving.redis_client.RedisClient.get_client')
def test_producer_consumer_flow(mock_get_client, fake_redis):
    #mock the redis client for both producer and consumer
    mock_get_client.return_value = fake_redis
    
    producer = Producer()
    consumer = Consumer()
    
    #create a dummy tensor (batch size 1)
    dummy_tensor = torch.randn(1, 3, 32, 32)
    
    #test producer pushes to queue
    req_id = producer.push(dummy_tensor)
    
    assert req_id is not None
    assert fake_redis.llen("inference_queue") == 1
    
    #test consumer pops from queue (non-blocking)
    req = consumer.pop_non_blocking()
    
    assert req is not None
    assert req["id"] == req_id
    
    #verify tensor shape was preserved through json serialization
    assert len(req["data"]) == 1
    assert len(req["data"][0]) == 3
    assert len(req["data"][0][0]) == 32
    
    #queue should be empty now
    assert fake_redis.llen("inference_queue") == 0
    assert consumer.pop_non_blocking() is None
