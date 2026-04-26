import json
import time
import uuid
from src.serving.redis_client import RedisClient

#redis queue key
QUEUE_NAME="inference_queue"

#pushes requests to the queue
class Producer:
  def __init__(self):
    self.redis=RedisClient().get_client()

  def push(self,tensor):
    req_id=str(uuid.uuid4())

    request={
      "id":req_id,
      "data":tensor.tolist(),
      "enqueue_time":time.time()
    }

    #push to left of queue
    self.redis.lpush(QUEUE_NAME,json.dumps(request))

    return req_id