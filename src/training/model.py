import torch
import torchvision.models as models

def get_model():
  model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)  #load the pretrained model mobilenet_2
  model.classifier[1] = torch.nn.Linear(model.last_channel,10)  #replace the last layer of the model with a new linear layer that has 10 output features (for 10 classes in CIFAR-10)
  return model