import time
import torch
import numpy as np
from src.inference.predictor import Predictor

predictor=Predictor("models/exported/model.pt") #load the TorchScript model
dummy=torch.randn(1,3,32,32)                    #dummy input for testing (batch size 1, 3 color channels, 32x32 pixels)

times=[]

for _ in range(10):
  predictor.predict(dummy)
  
#run inference 50 times and record the time taken for each prediction to calculate average latency
for _ in range(100):
  start=time.time()
  predictor.predict(dummy)
  end=time.time()
  times.append((end-start)*1000) #convert to milliseconds

print("Avg Latency:",round(np.mean(times),2),"ms")
print("Min Latency:",round(np.min(times),2),"ms")
print("Max Latency:",round(np.max(times),2),"ms")