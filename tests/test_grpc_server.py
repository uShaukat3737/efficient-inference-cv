import asyncio
import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
from PIL import Image

from src.grpc_api import inference_pb2
from src.grpc_api.server import InferenceServicer


def create_dummy_jpeg():
    img = Image.new("RGB", (32, 32), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestInferenceServicer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_producer = MagicMock()
        self.mock_redis = MagicMock()
        self.mock_producer.push.return_value = "test-uuid-123"
        self.mock_redis.get.return_value = "5"  # class_id 5 = "dog"

        self.servicer = InferenceServicer(
            producer=self.mock_producer,
            redis_client=self.mock_redis,
        )
        self.mock_context = AsyncMock()

    async def test_predict_returns_correct_class(self):
        request = inference_pb2.PredictRequest(image_data=create_dummy_jpeg())
        response = await self.servicer.Predict(request, self.mock_context)

        assert response.class_id == 5
        assert response.class_name == "dog"
        assert response.latency_ms > 0

    async def test_predict_calls_producer_push(self):
        request = inference_pb2.PredictRequest(image_data=create_dummy_jpeg())
        await self.servicer.Predict(request, self.mock_context)

        self.mock_producer.push.assert_called_once()

    async def test_predict_polls_redis_result_key(self):
        request = inference_pb2.PredictRequest(image_data=create_dummy_jpeg())
        await self.servicer.Predict(request, self.mock_context)

        self.mock_redis.get.assert_called_with("result:test-uuid-123")

    async def test_predict_invalid_image_aborts(self):
        request = inference_pb2.PredictRequest(image_data=b"not_an_image")
        await self.servicer.Predict(request, self.mock_context)

        self.mock_context.abort.assert_called_once()
        code = self.mock_context.abort.call_args[0][0]
        assert code == grpc.StatusCode.INVALID_ARGUMENT

    async def test_predict_timeout_aborts(self):
        self.mock_redis.get.return_value = None  # never returns a result

        with patch("src.grpc_api.server.POLL_ATTEMPTS", 1), \
             patch("src.grpc_api.server.POLL_INTERVAL", 0):
            request = inference_pb2.PredictRequest(image_data=create_dummy_jpeg())
            await self.servicer.Predict(request, self.mock_context)

        self.mock_context.abort.assert_called_once()
        code = self.mock_context.abort.call_args[0][0]
        assert code == grpc.StatusCode.DEADLINE_EXCEEDED


if __name__ == "__main__":
    unittest.main()
