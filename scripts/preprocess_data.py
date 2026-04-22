import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import json

#Load raw CIFAR-10 dataset, apply transformations, and save processed batches.
def preprocess_cifar10(batch_size=32):
  
  #define output directory
  output_dir = Path('data/processed')
  output_dir.mkdir(parents=True, exist_ok=True)
  
  print(f"Using device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}") #use mac mps
  
  #define transformations
  transform = transforms.Compose([
      transforms.ToTensor(),                #convert to tensor [0, 1]
      transforms.Normalize(
          mean=[0.4914, 0.4822, 0.4465],    #CIFAR-10 mean
          std=[0.2470, 0.2435, 0.2616]      #CIFAR-10 std
      )
  ])
  
  #split into 5:1 train:test
  #datasets
  trainset = torchvision.datasets.CIFAR10(
    root='data/raw',
    train=True,
    download=True,
    transform=transform
  )

  testset = torchvision.datasets.CIFAR10(
    root='data/raw',
    train=False,
    download=True,
    transform=transform
  )

  #dataloaders (batch processing)
  train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False
  )

  test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False
  )

  metadata = {
    'train': {'batch_sizes': [], 'cumulative_sizes': [0]},
    'test': {'batch_sizes': [], 'cumulative_sizes': [0]}
  }

  print("Processing training data in batches...")
  for i, (images, labels) in enumerate(train_loader):
      try:
          torch.save(images, output_dir / f"train_data_batch_{i}.pt")
          torch.save(labels, output_dir / f"train_labels_batch_{i}.pt")

          if i == 0:
              #save sample for inference benchmarking
              torch.save(images[:1], output_dir / "sample_input.pt")

          metadata['train']['batch_sizes'].append(len(images))
          metadata['train']['cumulative_sizes'].append(metadata['train']['cumulative_sizes'][-1] + len(images))

          if i % 10 == 0:
              print(f"  Saved train batch {i}")
      except Exception as e:
          print(f"Error saving train batch {i}: {e}")
          raise

  print("Processing test data in batches...")
  for i, (images, labels) in enumerate(test_loader):
      try:
          torch.save(images, output_dir / f"test_data_batch_{i}.pt")
          torch.save(labels, output_dir / f"test_labels_batch_{i}.pt")

          metadata['test']['batch_sizes'].append(len(images))
          metadata['test']['cumulative_sizes'].append(metadata['test']['cumulative_sizes'][-1] + len(images))

          if i % 10 == 0:
              print(f"  Saved test batch {i}")
      except Exception as e:
          print(f"Error saving test batch {i}: {e}")
          raise
          
  try:
      with open(output_dir / "metadata.json", "w") as f:
          json.dump(metadata, f, indent=4)
  except Exception as e:
      print(f"Error saving metadata.json: {e}")
      raise

  print("\n✓ Preprocessing completed (batch-based)")
  print(f"Location: {output_dir.resolve()}")

if __name__ == "__main__":
  preprocess_cifar10()
