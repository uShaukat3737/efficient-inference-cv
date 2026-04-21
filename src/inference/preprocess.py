from PIL import Image                         #get the standard Python image object
import torchvision.transforms as transforms

transform=transforms.Compose([
  transforms.Resize((32,32)),                 #resize the input image to 32x32 pixels (the size expected by the model) [hard constraint]
  transforms.ToTensor()                       #convert to pytorch tensor and bring pixel values to [0,1] (from original [0,255])
])

def preprocess(image:Image.Image):
  return transform(image).unsqueeze(0)        #apply transformation defined above and add an extra dimension at the beginning to represent the batch size (1 in this case, since we're processing a single image)

#(note : model expects input of shape [batch_size, channels, height, width], so we need to add the batch dimension even for a single image)