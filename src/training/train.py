import torch # import pytorch
import torchvision # get datasets and models
import torchvision.transforms as transforms # module for data transformations [converting images to tensors]
from model import get_model # load our model from model.py

def train():

  #set device to MPS if available, otherwise CPU (too slow on cpu)
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  print(f"Using device: {device}")

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
  model = get_model().to(device)
  criterion = torch.nn.CrossEntropyLoss() #penelty for incorrect predictions
  optimizer = torch.optim.Adam(model.parameters(),lr = 0.001) #update model parameters based on the computed gradients

  #train the model for 5 epochs initially
  print(f"Training for {5} epochs")
  for epoch in range(5):

    print(f"Epoch {epoch+1}/{5}")
    running_loss = 0.0

    for images,labels in trainloader: #get the batch and iterate over it
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)           #forward pass
      loss = criterion(outputs,labels)  #get the loss between prediction and true labels

      running_loss += loss.item()

      #get ready for the next batch
      optimizer.zero_grad()             #clear previous gradients       
      loss.backward()                   #calculate gradients for every weight in network
      optimizer.step()                  #update weights based on calculated gradients (get gradient -> mulity by learning rate -> subtract from current weights) (move towards minimizing the loss)

    print(f"Epoch {epoch+1} completed. Average Loss: {running_loss / len(trainloader):.4f}")

  #save the trained model's parameters (learned weights only) to a file for later use
  torch.save(model.state_dict(),"models/trained/mobilenetv2.pth")

  print("Training completed.")

#only runs when executed directly, not when imported as a module
if __name__ == "__main__":
  train()