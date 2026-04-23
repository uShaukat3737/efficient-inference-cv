import json
from src.serving.redis_client import RedisClient

#redis queue key
QUEUE_NAME="inference_queue"

#pulls requests from the queue
class Consumer:
  def __init__(self):
    self.redis=RedisClient().get_client()

  def pop_blocking(self):
    #blocks indefinitely until an item arrives
    result = self.redis.brpop(QUEUE_NAME, timeout=0)
    if result:
      _,data=result
      return json.loads(data)
    return None

  def pop_non_blocking(self):
    #pops immediately without blocking
    data = self.redis.rpop(QUEUE_NAME)
    if data:
      return json.loads(data)
    return None