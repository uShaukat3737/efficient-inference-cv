from PIL import Image                         #get the standard Python image object
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)

transform=transforms.Compose([
  transforms.Resize((32,32)),                 #resize the input image to 32x32 pixels (the size expected by the model) [hard constraint]
  transforms.ToTensor(),                      #convert to pytorch tensor and bring pixel values to [0,1] (from original [0,255])
  transforms.Normalize(
      mean=[0.4914, 0.4822, 0.4465],         #CIFAR-10 mean (must match training normalization)
      std=[0.2470, 0.2435, 0.2616]           #CIFAR-10 std (must match training normalization)
  )
])

def preprocess(image:Image.Image):
  try:
    return transform(image).unsqueeze(0)        #apply transformation defined above and add an extra dimension at the beginning to represent the batch size (1 in this case, since we're processing a single image)
  except Exception as e:
    logger.error(f"Image preprocessing failed: {e}")
    raise ValueError(f"Image preprocessing failed: {e}")

#(note : model expects input of shape [batch_size, channels, height, width], so we need to add the batch dimension even for a single image)