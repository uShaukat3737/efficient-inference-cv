import torch # import pytorch
import torchvision # get datasets and models
import torchvision.transforms as transforms # module for data transformations [converting images to tensors]
from model import get_model # load our model from model.py

def train():
  #convert images to tensors and normalize them
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  #load the CIFAR-10 training dataset, applying the transformations defined above
  trainset = torchvision.datasets.CIFAR10(
    root = 'data/raw', train = True, download = True, transform = transform)

  #split into batches of 32 and shuffle the data for better training
  trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle = True)

  #initialize the model, loss function, and optimizer
  model = get_model()
  criterion = torch.nn.CrossEntropyLoss() #penelty for incorrect predictions
  optimizer = torch.optim.Adam(model.parameters(),lr = 0.001) #update model parameters based on the computed gradients

  #train the model for 5 epochs initially
  for epoch in range(5):
    
    for images,labels in trainloader: #get the batch and iterate over it
      outputs = model(images)           #forward pass
      loss = criterion(outputs,labels)  #get the loss between prediction and true labels

      #get ready for the next batch
      optimizer.zero_grad()             #clear previous gradients       
      loss.backward()                   #calculate gradients for every weight in network
      optimizer.step()                  #update weights based on calculated gradients (get gradient -> mulity by learning rate -> subtract from current weights) (move towards minimizing the loss)

  #save the trained model's parameters (learned weights only) to a file for later use
  torch.save(model.state_dict(),"models/trained/mobilenetv2.pth")

#only runs when executed directly, not when imported as a module
if __name__ == "__main__":
  train()