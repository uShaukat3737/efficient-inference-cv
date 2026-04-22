import torch
import torchvision
from pathlib import Path
import logging
import json
import functools

# Import model architecture (handles both module and script execution)
try:
  from .model import get_model
except ImportError:
  from model import get_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreprocessedCIFAR10(torch.utils.data.Dataset):
  """Load preprocessed CIFAR-10 batches from saved tensors with lazy loading."""
  
  def __init__(self, data_dir='data/processed', train=True):
    """Initialize dataset with lazy loading (batches loaded per index, not all at once)."""
    self.data_dir = Path(data_dir)
    self.train = train
    self.split = 'train' if train else 'test'
    suffix = self.split
    
    # Find batch files
    self.batch_files = sorted(self.data_dir.glob(f'{suffix}_data_batch_*.pt'))
    self.label_files = sorted(self.data_dir.glob(f'{suffix}_labels_batch_*.pt'))
    
    if not self.batch_files:
      raise FileNotFoundError(f"No batch files found in {self.data_dir}")
    
    if len(self.batch_files) != len(self.label_files):
      raise ValueError(f"Mismatch: {len(self.batch_files)} data batches, {len(self.label_files)} label batches")
    
    # Track cumulative sizes for lazy loading
    self.batch_sizes = []
    self.cumulative_sizes = [0]
    
    # Try to load metadata
    metadata_file = self.data_dir / 'metadata.json'
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.batch_sizes = metadata[self.split]['batch_sizes']
            self.cumulative_sizes = metadata[self.split]['cumulative_sizes']
            logger.info("Loaded dataset sizes from metadata.json")
        except Exception as e:
            logger.warning(f"Failed to load metadata.json: {e}. Falling back to slow file loading.")
            
    if not self.batch_sizes:
        logger.info("metadata.json not found or failed, scanning files (this may be slow)...")
        for batch_file in self.batch_files:
          try:
            batch_data = torch.load(batch_file)
            self.batch_sizes.append(len(batch_data))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(batch_data))
          except Exception as e:
            raise RuntimeError(f"Failed to load batch file {batch_file}: {e}")
    
    logger.info(f"Loaded {len(self.batch_files)} batches, total samples: {self.cumulative_sizes[-1]}")
  
  def __len__(self):
    return self.cumulative_sizes[-1]
    
  @functools.lru_cache(maxsize=16)
  def _load_batch(self, batch_idx):
    """Load and cache a specific batch using LRU cache."""
    try:
      batch_data = torch.load(self.batch_files[batch_idx])
      batch_labels = torch.load(self.label_files[batch_idx])
      return batch_data, batch_labels
    except Exception as e:
      raise RuntimeError(f"Failed to load batch {batch_idx}: {e}")

  def __getitem__(self, idx):
    """Lazy load data: only load the specific batch needed for this index."""
    if idx < 0 or idx >= len(self):
      raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
    
    # Find which batch contains this index
    batch_idx = 0
    for i, cumulative_size in enumerate(self.cumulative_sizes[1:], 1):
      if idx < cumulative_size:
        batch_idx = i - 1
        break
    
    idx_within_batch = idx - self.cumulative_sizes[batch_idx]
    
    batch_data, batch_labels = self._load_batch(batch_idx)
    return batch_data[idx_within_batch], batch_labels[idx_within_batch]

def train():
  """Train the model with error handling and logging."""
  try:
    # Use MPS if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    #load the preprocessed CIFAR-10 training dataset
    try:
      trainset = PreprocessedCIFAR10(train=True)
      logger.info(f"Loaded training dataset with {len(trainset)} samples")
    except FileNotFoundError as e:
      logger.error(f"Dataset not found: {e}. Run preprocess_data.py first.")
      raise
    except Exception as e:
      logger.error(f"Failed to load dataset: {e}")
      raise

    # Create dataloader with batch size 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    # Initialize model, loss, optimizer
    try:
      model = get_model().to(device)
      logger.info("Model loaded successfully")
    except Exception as e:
      logger.error(f"Failed to load model: {e}")
      raise
    
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create checkpoint directory
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train for 5 epochs
    num_epochs = 5
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
      logger.info(f"Epoch {epoch+1}/{num_epochs}")
      running_loss = 0.0

      try:
        for batch_idx, (images, labels) in enumerate(trainloader):
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          loss = criterion(outputs, labels)
          running_loss += loss.item()

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      except Exception as e:
        logger.error(f"Error during training batch {batch_idx}: {e}")
        raise

      avg_loss = running_loss / len(trainloader)
      logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
      
      #save checkpoint after each epoch
      try:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path.name}")
      except Exception as e:
        logger.error(f"Failed to save checkpoint at epoch {epoch+1}: {e}")
        raise

    # Save trained model
    try:
      model_save_path = Path('models/trained/mobilenetv2.pth')
      model_save_path.parent.mkdir(parents=True, exist_ok=True)
      torch.save(model.state_dict(), model_save_path)
      logger.info(f"Model saved to {model_save_path}")
    except Exception as e:
      logger.error(f"Failed to save model: {e}")
      raise

    logger.info("Training completed successfully.")
  
  except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

# Run training
if __name__ == "__main__":
  train()