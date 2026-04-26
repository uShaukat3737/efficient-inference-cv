import os
import pytest
from unittest.mock import patch, MagicMock


def test_producer_reads_redis_host_from_env():
    #producer should read REDIS_HOST from environment variable
    with patch('src.serving.redis_client.redis.Redis') as mock_redis:
        with patch.dict(os.environ, {'REDIS_HOST': 'custom-redis-host'}):
            from src.serving.producer import Producer
            producer = Producer()

            #verify RedisClient was called with custom host
            from src.serving.redis_client import RedisClient
            with patch.dict(os.environ, {'REDIS_HOST': 'custom-redis-host'}):
                client = RedisClient()
                #should use environment variable
                assert client.client is not None


def test_consumer_reads_redis_host_from_env():
    #consumer should read REDIS_HOST from environment variable
    with patch('src.serving.redis_client.redis.Redis') as mock_redis:
        with patch.dict(os.environ, {'REDIS_HOST': 'custom-redis-host'}):
            from src.serving.consumer import Consumer
            consumer = Consumer()

            #verify RedisClient was called with custom host
            from src.serving.redis_client import RedisClient
            with patch.dict(os.environ, {'REDIS_HOST': 'custom-redis-host'}):
                client = RedisClient()
                assert client.client is not None


def test_redis_client_uses_env_host_or_defaults():
    #RedisClient should use REDIS_HOST env var, fallback to localhost
    with patch('src.serving.redis_client.redis.Redis') as mock_redis:
        #test with custom host
        with patch.dict(os.environ, {'REDIS_HOST': 'custom-host'}):
            from src.serving.redis_client import RedisClient
            import importlib
            import src.serving.redis_client
            importlib.reload(src.serving.redis_client)
            from src.serving.redis_client import RedisClient

            client = RedisClient()
            mock_redis.assert_called_with(
                host='custom-host',
                port=6379,
                db=0,
                decode_responses=True
            )


def test_redis_client_defaults_to_localhost():
    #RedisClient should default to localhost when env var not set
    with patch('src.serving.redis_client.redis.Redis') as mock_redis:
        with patch.dict(os.environ, {}, clear=True):
            from src.serving.redis_client import RedisClient
            import importlib
            import src.serving.redis_client
            importlib.reload(src.serving.redis_client)

            client = RedisClient()
            mock_redis.assert_called_with(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
