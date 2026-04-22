from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import logging
from src.inference.predictor import Predictor
from src.inference.preprocess import preprocess
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CIFAR-10 Inference API")

# Model loaded on startup
predictor = None

@app.on_event("startup")
async def startup_event():
  #Load model on API startup.
  global predictor
  try:
    predictor = Predictor("models/exported/model.pt")
    logger.info("Model loaded successfully on startup (TorchScript)")
  except Exception as e_pt:
    logger.warning(f"Failed to load model.pt: {e_pt}. Trying model.onnx...")
    try:
      predictor = Predictor("models/exported/model.onnx")
      logger.info("Model loaded successfully on startup (ONNX)")
    except Exception as e_onnx:
      logger.warning(f"Failed to load model.onnx: {e_onnx}. Trying mobilenetv2.pth...")
      try:
        predictor = Predictor("models/trained/mobilenetv2.pth")
        logger.info("Model loaded successfully on startup (PyTorch state_dict)")
      except Exception as e_pth:
        logger.error(f"All model loading attempts failed. Last error: {e_pth}")
        raise RuntimeError("Failed to load any model on startup")

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
  #Predict image class. Accepts JPEG, PNG formats only.
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
      #force RGB to avoid RGBA/grayscale issues
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
    
    #preprocess and predict
    try:
      input_tensor = preprocess(image)
      pred = predictor.predict(input_tensor)
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

@app.get("/health")
async def health_check():
  #health check endpoint.
  try:
    return {"status": "healthy", "model_loaded": predictor is not None}
  except Exception as e:
    logger.error(f"Health check failed: {e}")
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail="Health check failed"
    )