import torch
import onnxruntime as ort
import numpy as np
from src.training.model import get_model

class Predictor:
  def __init__(self, model_path):
    self.model_type = None
    self.model = None
    self.session = None
    
    if model_path.endswith('.pt'):
      # Load TorchScript model
      self.model = torch.jit.load(model_path)
      self.model.eval()
      self.model_type = 'torchscript'
    elif model_path.endswith('.pth'):
      # Load raw PyTorch model
      self.model = get_model()
      self.model.load_state_dict(torch.load(model_path, weights_only=False))
      self.model.eval()
      self.model_type = 'pytorch'
    elif model_path.endswith('.onnx'):
      # Load ONNX model
      self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
      self.model_type = 'onnx'
    else:
      raise ValueError(f"Unsupported model format: {model_path}")

  # classify the image 
  def predict(self, input_tensor):
    if self.model_type == 'onnx':
      # Handle ONNX prediction
      if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.numpy()
      
      input_name = self.session.get_inputs()[0].name
      output = self.session.run(None, {input_name: input_tensor})
      output_tensor = torch.tensor(output[0])
      return output_tensor.argmax(dim=1).item()
    else:
      # Handle PyTorch models (TorchScript and raw PyTorch)
      with torch.no_grad(): #not training data so better to not calculate gradients (saves memory and computations)
        output = self.model(input_tensor) #forward pass through the model to get the output predictions (logits) for each class
        return output.argmax(dim=1).item() #pick the class with the highest predicted score (logit) and return it as the predicted class index (convert from tensor to a regular Python integer)