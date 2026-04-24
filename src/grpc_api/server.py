import asyncio
import io
import logging
import time

import grpc
import torch
from grpc import aio
from PIL import Image, UnidentifiedImageError

from src.grpc_api import inference_pb2, inference_pb2_grpc
from src.inference.preprocess import preprocess
from src.serving.producer import Producer

# Mirror the FastAPI server's thread-limiting decision — prevents CPU thrashing
# under concurrent gRPC calls sharing the same process.
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

POLL_ATTEMPTS = 50
POLL_INTERVAL = 0.1


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, producer=None, redis_client=None):
        self._producer = producer or Producer()
        self._redis = redis_client or self._producer.redis

    async def Predict(self, request, context):
        t0 = time.perf_counter()

        try:
            image = Image.open(io.BytesIO(request.image_data)).convert("RGB")
        except (UnidentifiedImageError, Exception) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid image: {exc}")
            return inference_pb2.PredictResponse()

        try:
            tensor = preprocess(image)
            req_id = self._producer.push(tensor)
        except Exception as exc:
            logger.error("Preprocessing or queue push failed: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Preprocessing failed")
            return inference_pb2.PredictResponse()

        result_key = f"result:{req_id}"
        for _ in range(POLL_ATTEMPTS):
            raw = self._redis.get(result_key)
            if raw is not None:
                class_id = int(raw)
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.info("Prediction successful: class %d in %.1fms", class_id, latency_ms)
                return inference_pb2.PredictResponse(
                    class_id=class_id,
                    class_name=CIFAR10_CLASSES[class_id],
                    latency_ms=latency_ms,
                )
            await asyncio.sleep(POLL_INTERVAL)

        await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Inference timed out")
        return inference_pb2.PredictResponse()


async def serve(port: int = 50051) -> None:
    server = aio.server()
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info("gRPC server started on port %d", port)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
