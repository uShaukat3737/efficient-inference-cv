import torch
import onnxruntime as ort
import numpy as np
import logging
from pathlib import Path
from src.training.model import get_model

logger = logging.getLogger(__name__)

class Predictor:
  
  def __init__(self, model_path: str, device: str = "cpu"):

    self.model_type = None
    self.model = None
    self.session = None
    self.device = torch.device(device)

    model_path = Path(model_path)
    if not model_path.exists():
      raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
      if model_path.suffix == '.pt':
        #load TorchScript model
        try:
          self.model = torch.jit.load(model_path)
          self.model.eval()
          try:
            self.model = self.model.to(self.device)
          except Exception:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
          self.model_type = 'torchscript'
          logger.info(f"Loaded TorchScript model from {model_path} on {self.device}")
        except Exception as e:
          raise RuntimeError(f"Failed to load TorchScript model: {e}")

      elif model_path.suffix == '.pth':
        #load raw PyTorch model
        try:
          self.model = get_model()
          state_dict = torch.load(model_path, weights_only=False)
          self.model.load_state_dict(state_dict)
          self.model.eval()
          try:
            self.model = self.model.to(self.device)
          except Exception:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
          self.model_type = 'pytorch'
          logger.info(f"Loaded PyTorch model from {model_path} on {self.device}")
        except Exception as e:
          raise RuntimeError(f"Failed to load PyTorch model: {e}")

      elif model_path.suffix == '.onnx':
        #load ONNX model; ONNX Runtime has no MPS backend, always uses CPU
        try:
          if device != "cpu":
            logger.warning(f"ONNX Runtime has no MPS backend. Ignoring device='{device}', using CPU.")
          self.device = torch.device("cpu")
          sess_options = ort.SessionOptions()
          self.session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=['CPUExecutionProvider']
          )
          self.model_type = 'onnx'
          logger.info(f"Loaded ONNX model from {model_path}")
        except Exception as e:
          raise RuntimeError(f"Failed to load ONNX model: {e}")
      else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}. Supported: .pt, .pth, .onnx")
    
    except (FileNotFoundError, ValueError, RuntimeError) as e:
      logger.error(f"Model loading error: {e}")
      raise

  def predict(self, input_tensor: torch.Tensor) -> int:
    try:
      if self.model_type == 'onnx':
        #handle ONNX prediction
        try:
          if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.cpu().numpy()

          if not isinstance(input_tensor, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(input_tensor)}")

          input_name = self.session.get_inputs()[0].name
          output = self.session.run(None, {input_name: input_tensor})
          output_tensor = torch.tensor(output[0])
          prediction = output_tensor.argmax(dim=1).item()
          return prediction
        except Exception as e:
          raise RuntimeError(f"ONNX inference failed: {e}")

      else:
        #handle PyTorch models (TorchScript and raw PyTorch)
        try:
          if not isinstance(input_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(input_tensor)}")

          input_tensor = input_tensor.to(self.device)
          with torch.no_grad():
            output = self.model(input_tensor)
            prediction = output.argmax(dim=1).item()
            return prediction
        except Exception as e:
          raise RuntimeError(f"PyTorch inference failed: {e}")
    
    except RuntimeError as e:
      logger.error(f"Prediction error: {e}")
      raise

  def predict_batch(self, batch_tensor: torch.Tensor) -> list[int]:
    try:
      if self.model_type == 'onnx':
        #handle ONNX prediction
        try:
          if isinstance(batch_tensor, torch.Tensor):
            batch_tensor = batch_tensor.cpu().numpy()

          if not isinstance(batch_tensor, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(batch_tensor)}")

          input_name = self.session.get_inputs()[0].name
          output = self.session.run(None, {input_name: batch_tensor})
          output_tensor = torch.tensor(output[0])
          predictions = output_tensor.argmax(dim=1).tolist()
          return predictions
        except Exception as e:
          raise RuntimeError(f"ONNX batch inference failed: {e}")

      else:
        #handle PyTorch models (TorchScript and raw PyTorch)
        try:
          if not isinstance(batch_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(batch_tensor)}")

          batch_tensor = batch_tensor.to(self.device)
          with torch.no_grad():
            output = self.model(batch_tensor)
            predictions = output.argmax(dim=1).tolist()
            return predictions
        except Exception as e:
          raise RuntimeError(f"PyTorch batch inference failed: {e}")
    
    except RuntimeError as e:
      logger.error(f"Batch prediction error: {e}")
      raise