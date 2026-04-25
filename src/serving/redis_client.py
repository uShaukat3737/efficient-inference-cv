import redis
import os

#redis client wrapper

class RedisClient:
  def __init__(self, host=None, port=6379, db=0):
    if host is None:
      host = os.getenv("REDIS_HOST", "localhost")
    self.client=redis.Redis(host=host,port=port,db=db,decode_responses=True)

  def get_client(self):
    return self.client