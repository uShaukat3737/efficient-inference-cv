import torch 
from src.training.model import get_model #import the function to get the model architecture
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model=get_model()

try:
    model.load_state_dict(torch.load("models/trained/mobilenetv2.pth", weights_only=False))
    logger.info("Successfully loaded trained model weights.")
except FileNotFoundError:
    logger.warning("Trained model weights not found at models/trained/mobilenetv2.pth! Proceeding to export UNTRAINED default model architecture as fallback.")
except Exception as e:
    logger.warning(f"Failed to load trained model weights ({e}). Proceeding to export UNTRAINED default model architecture as fallback.")

#set model to evaluation mode 
model.eval()

dummy_input=torch.randn(1,3,32,32) #(batch size 1, 3 color channels, 32x32 image size) (for tracing and exporting the model)

os.makedirs("models/exported", exist_ok=True)

try:
    #TorchScript
    ts_model=torch.jit.trace(model,dummy_input) #convert the model to TorchScript format by tracing its execution with the dummy input (converts model into static graph to run faster in production (no python dependency))
    ts_model.save("models/exported/model.pt")
    logger.info("Successfully exported TorchScript model to models/exported/model.pt")
except Exception as e:
    logger.error(f"Failed to export TorchScript model: {e}")

try:
    #ONNX
    torch.onnx.export(model,dummy_input,"models/exported/model.onnx") #also convert for onnx as industry standard 
    logger.info("Successfully exported ONNX model to models/exported/model.onnx")
except Exception as e:
    logger.error(f"Failed to export ONNX model: {e}")

#now we have PyTorch, TorchScript, and ONNX versions of the model for efficient inference in production