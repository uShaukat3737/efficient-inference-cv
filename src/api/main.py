from fastapi import FastAPI,UploadFile            #use Fastapi for routehandling
from PIL import Image
from src.inference.predictor import Predictor
from src.inference.preprocess import preprocess

app=FastAPI()

predictor=Predictor("models/exported/model.pt")   #load the TorchScript traced model to an instance of the Predictor class 

LABELS = [
  "airplane","automobile","bird","cat","deer",
  "dog","frog","horse","ship","truck"
]

@app.post("/predict")                             #on the /predict endpoint, upload image for prediction
async def predict(file:UploadFile):
  image=Image.open(file.file)                     #bring image to PLT image format
  input_tensor=preprocess(image)                  #preprocess the image (resize, convert to tensor, add batch dimension)  
  pred=predictor.predict(input_tensor)            #predict its class
  return {
    "class_id": int(pred),
    "class_name": LABELS[pred]
  }                 #return the predicted class index as a JSON response (convert from numpy int to regular Python int for better compatibility with JSON) (added: also return the class name for better readability)