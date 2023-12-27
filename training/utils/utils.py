import torch
from torch.nn.modules.loss import _Loss
from torch.optim import SGD
from torch.nn import Module
import os

def get_default_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() \
      else torch.device('cpu')

def to_device(data, device):
    return [to_device(x, device) for x in data] if isinstance(data, (list,tuple)) \
      else data.to(device, non_blocking=True)

def batches_to_device(data_loader, device):
    for batch in data_loader:
        yield to_device(batch, device)

def cuda(tensor):
  """
  Return tensor on cuda if cuda available
  """
  return tensor.cuda() if torch.cuda.is_available() else tensor

def save_checkpoints(epoch: int, model: Module, optimizer: SGD, loss: _Loss, path: str):
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, path)

def save_model(model: Module, model_name: str, dir: str):
  torch.save(model, f'{dir}{model_name}.pt')

def load_checkpoint(model: Module, checkpoint_path: str, label: str = 'model_state_dict'):
  if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path)[label])
  model.eval()

# def load_model(path: str):
#   model = torch.load(path)
#   if isinstance(model, dict):
#      model = model['model']
#   model.eval()
  
  # return model

def load_model(path: str, map_location: str = 'cpu'):
  model = torch.load(path, map_location=torch.device(map_location))
  if isinstance(model, dict):
    model = model['model']
  # model.eval()

  return model

def clear_memory():
  torch.cuda.empty_cache()

def unpack(batch):
    return (batch[0], batch[1], None) if len(batch) == 2 else batch


def mkdir(path):
    if not os.path.exists(path): os.makedirs(path)