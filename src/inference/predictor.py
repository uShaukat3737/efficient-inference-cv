import torch

class Predictor:
  def __init__(self,model_path):
    self.model=torch.jit.load(model_path) #load the TorchScript traced model from the specified path
    self.model.eval()                     #disable dropout and batch normalization layers for inference (evaluation mode)

#classify the image 
  def predict(self,input_tensor):         
    with torch.no_grad():                 #not training data so better to not calculate gradients (saves memory and computations)
      output=self.model(input_tensor)     #forward pass through the model to get the output predictions (logits) for each class
      return output.argmax(dim=1).item()  #pick the class with the highest predicted score (logit) and return it as the predicted class index (convert from tensor to a regular Python integer)