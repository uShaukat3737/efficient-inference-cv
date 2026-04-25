#tests/test_batching.py
import pytest
import torch
import time
from unittest.mock import MagicMock
from src.serving.batch_scheduler import BatchScheduler

#custom exception inheriting from BaseException so it bypasses the 'except Exception' block in the while loop
class StopLoop(BaseException):
    pass

class MockConsumer:
    def __init__(self, items):
        self.items = items
        self.redis = MagicMock()
        
class MockModel:
    def __init__(self):
        self.predict_batch_called = False
        self.last_batch_size = 0
        
    def predict_batch(self, batch_tensor):
        self.predict_batch_called = True
        self.last_batch_size = batch_tensor.shape[0]
        return [0] * self.last_batch_size

def test_batch_size_trigger():
    #create 8 dummy requests (a full batch)
    items = []
    for i in range(8):
        items.append({
            "id": f"req_{i}",
            "data": torch.randn(1, 3, 32, 32).tolist()
        })
        
    consumer = MockConsumer(items)
    model = MockModel()
    
    scheduler = BatchScheduler(consumer, model, batch_size=8, timeout=1.0)
    
    #mock the pop functions to return our items, and then violently stop the loop
    def exploding_pop():
        if consumer.items:
            return consumer.items.pop(0)
        raise StopLoop()
        
    consumer.pop_non_blocking = exploding_pop
    consumer.pop_blocking = exploding_pop
    
    try:
        scheduler.run()
    except StopLoop:
        pass
        
    #verify the model was called with exactly 8 items (the batch size trigger fired)
    assert model.predict_batch_called
    assert model.last_batch_size == 8

def test_timeout_trigger():
    #create only 1 dummy request
    items = [{
        "id": "req_0",
        "data": torch.randn(1, 3, 32, 32).tolist()
    }]
    
    consumer = MockConsumer(items)
    model = MockModel()
    
    #set a very short timeout
    scheduler = BatchScheduler(consumer, model, batch_size=8, timeout=0.01)
    
    def stalling_pop():
        if consumer.items:
            return consumer.items.pop(0)
            
        #simulate sleeping past the timeout so the loop is forced to process the partial batch
        time.sleep(0.02)
        
        #return None once to let the loop execute its timeout check
        #and replace ourselves with an exploding pop for the NEXT loop iteration
        def thrower(): raise StopLoop()
        consumer.pop_non_blocking = thrower
        consumer.pop_blocking = thrower
        return None
        
    consumer.pop_non_blocking = stalling_pop
    consumer.pop_blocking = stalling_pop
    
    try:
        scheduler.run()
    except StopLoop:
        pass
        
    #verify the model was called with exactly 1 item (the timeout trigger fired)
    assert model.predict_batch_called
    assert model.last_batch_size == 1


def test_metrics_written_to_redis():
    #verify that metrics:{req_id} is written to Redis with correct structure
    items = []
    for i in range(2):
        items.append({
            "id": f"req_{i}",
            "data": torch.randn(1, 3, 32, 32).tolist(),
            "enqueue_time": time.time()
        })

    consumer = MockConsumer(items)
    model = MockModel()
    #batch_size=8 (won't trigger), timeout=0.01 (will trigger after batch of 2)
    scheduler = BatchScheduler(consumer, model, batch_size=8, timeout=0.01)

    def stalling_pop():
        if consumer.items:
            return consumer.items.pop(0)
        #sleep long enough to trigger timeout
        time.sleep(0.02)
        def thrower():
            raise StopLoop()
        consumer.pop_non_blocking = thrower
        consumer.pop_blocking = thrower
        return None

    consumer.pop_non_blocking = stalling_pop
    consumer.pop_blocking = stalling_pop

    try:
        scheduler.run()
    except StopLoop:
        pass

    #verify that setex was called for both result: and metrics: keys
    assert consumer.redis.setex.called, "setex was never called"
    call_count = consumer.redis.setex.call_count
    #2 requests × 2 setex calls (result + metrics) = 4 total
    assert call_count >= 4, f"Expected at least 4 setex calls (2 results + 2 metrics), got {call_count}"

    #verify metrics:{req_id} keys are present
    metrics_keys = [call[0][0] for call in consumer.redis.setex.call_args_list]
    metrics_keys_matching = [k for k in metrics_keys if k.startswith("metrics:")]
    assert len(metrics_keys_matching) == 2, f"Expected 2 metrics keys, found {metrics_keys_matching}"
