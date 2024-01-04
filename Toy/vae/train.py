
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):

    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step()
