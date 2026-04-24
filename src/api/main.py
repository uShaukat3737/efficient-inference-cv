from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import logging
import asyncio
from src.serving.producer import Producer
from src.inference.preprocess import preprocess
from src.inference.predictor import Predictor
import io
import torch
from contextlib import asynccontextmanager

# Limit PyTorch threads to 1 to avoid CPU thrashing during concurrent requests
torch.set_num_threads(1)

#setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#app initialized below after lifespan definition

#producer initialized on startup
producer = None
sync_predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
  #initialize redis producer on startup
  global producer, sync_predictor
  try:
    producer = Producer()
    #verify connection
    producer.redis.ping()
    logger.info("Successfully connected to Redis and initialized Producer")
  except Exception as e:
    logger.error(f"Failed to initialize Redis Producer: {e}")
    raise RuntimeError("Failed to connect to Redis on startup")

  try:
    sync_predictor = Predictor("models/exported/model.pt")
    logger.info("Successfully loaded synchronous Predictor")
  except Exception as e:
    logger.error(f"Failed to load synchronous Predictor: {e}")
    
  yield
  #cleanup could go here when the app shuts down

app = FastAPI(title="CIFAR-10 Inference API", lifespan=lifespan)

LABELS = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck"
]

@app.post("/predict", responses={
  200: {"description": "Successful prediction"},
  400: {"description": "Invalid image format"},
  500: {"description": "Server error"}
})
async def predict(file: UploadFile):
  #predict image class. accepts jpeg/png formats only
  try:
    #validate file type
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
      logger.warning(f"Invalid file type: {file.content_type}")
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid file type: {file.content_type}. Allowed: JPEG, PNG"
      )
    
    #read and open image
    try:
      image_data = await file.read()
      if not image_data:
        raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail="Empty file uploaded"
        )
      
      image = Image.open(io.BytesIO(image_data))
      #force rgb to avoid rgba/grayscale issues
      if image.mode != 'RGB':
        image = image.convert('RGB')
      logger.info(f"Image opened successfully: {image.size}, mode: {image.mode}")
    except UnidentifiedImageError as e:
      logger.error(f"Unidentified image format: {e}")
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="The uploaded file is not a valid image format that can be identified."
      )
    except ValueError as e:
      logger.error(f"Value error when opening image: {e}")
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid image content: {str(e)}"
      )
    except Exception as e:
      logger.error(f"Failed to open image: {e}")
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid image format: {str(e)}"
      )
    
    #preprocess and queue prediction
    try:
      input_tensor = preprocess(image)
      
      #enqueue the request
      req_id = producer.push(input_tensor)
      
      #async polling for the result
      result_key = f"result:{req_id}"
      pred = None
      
      #poll for up to 5 seconds (50 * 0.1s)
      for _ in range(50):
          result = producer.redis.get(result_key)
          if result is not None:
              pred = int(result)
              break
          await asyncio.sleep(0.1)
          
      if pred is None:
          raise TimeoutError("Inference request timed out waiting for worker")
          
      logger.info(f"Prediction successful: class {pred}")
    except Exception as e:
      logger.error(f"Prediction failed: {e}")
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Prediction failed: {str(e)}"
      )
    
    return {
      "class_id": int(pred),
      "class_name": LABELS[pred],
      "confidence": "N/A (model output logits, not probabilities)"
    }
  
  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Unexpected error in predict endpoint: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail="Internal server error"
    )

#baseline model without using redis or async
@app.post("/predict_sync")
def predict_sync(file: UploadFile):
  try:
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
      raise HTTPException(status_code=400, detail="Invalid file type")
      
    image_data = file.file.read()
    image = Image.open(io.BytesIO(image_data))
    if image.mode != 'RGB':
      image = image.convert('RGB')
      
    input_tensor = preprocess(image)
    
    #run direct blocking inference on the main thread pool
    predictions = sync_predictor.predict_batch(input_tensor)
    pred = int(predictions[0])
    
    return {
      "class_id": pred,
      "class_name": LABELS[pred],
      "confidence": "N/A (Sync API)"
    }
  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"Sync prediction failed: {e}")
    raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
  #health check endpoint
  try:
    redis_connected = producer.redis.ping()
    return {"status": "healthy", "redis_connected": redis_connected}
  except Exception as e:
    logger.error(f"Health check failed: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail="Health check failed"
    )