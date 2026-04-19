import torch 
from src.training.model import get_model #import the function to get the model architecture

model=get_model()
model.load_state_dict(torch.load("models/trained/mobilenetv2.pth"))  #load the trained model's parameters (learned weights) from the file into the model architecture

#set model to evaluation mode 
model.eval()

dummy_input=torch.randn(1,3,32,32) #(batch size 1, 3 color channels, 32x32 image size) (for tracing and exporting the model)

#TorchScript
ts_model=torch.jit.trace(model,dummy_input) #convert the model to TorchScript format by tracing its execution with the dummy input (converts model into static graph to run faster in production (no python dependency))
ts_model.save("models/exported/model.pt")

#ONNX
torch.onnx.export(model,dummy_input,"models/exported/model.onnx") # also convert for onnx as industry standard 

#now we have PyTorch, TorchScript, and ONNX versions of the model for efficient inference in production