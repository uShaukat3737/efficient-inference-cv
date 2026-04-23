import torch
from src.serving.consumer import Consumer
from src.serving.batch_scheduler import BatchScheduler
from src.inference.predictor import Predictor

#load your TorchScript / model
MODEL_PATH="models/exported/model.pt"

def main():
  print("Loading model...")
  #unified predictor handles torchscript/onnx
  predictor=Predictor(MODEL_PATH)

  consumer=Consumer()

  scheduler=BatchScheduler(
    consumer=consumer,
    model=predictor,
    batch_size=8,
    timeout=0.02
  )

  print("Worker started... Listening for requests")
  scheduler.run()

if __name__=="__main__":
  main()