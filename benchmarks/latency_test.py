import time
import torch
from src.inference.predictor import Predictor

predictor=Predictor("models/exported/model.pt") #load the TorchScript model

dummy=torch.randn(1,3,32,32)                    #dummy input for testing (batch size 1, 3 color channels, 32x32 pixels)

times=[]

#run inference 50 times and record the time taken for each prediction to calculate average latency
for _ in range(50):
  start=time.time()
  predictor.predict(dummy)
  end=time.time()
  times.append(end-start)

print("Avg Latency:",sum(times)/len(times))