import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from model.base import BaseVAE
from torchvision.utils import make_grid


def train_step(model: BaseVAE,
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    total_loss = {
        'loss': 0,
        'Reconstruction_Loss': 0,
        'KLD': 0,
    }
    for batch_idx, (image, label) in enumerate(dataloader):
        # image, label = image.to(device), label.to(device)
        print(f'\r{batch_idx}/{len(dataloader)}', end='', flush=True)
        image = image.to(device)
        results = model(image)
        loss = model.loss_function(*results,
                                   M_N=0.00025)

        total_loss['loss'] += loss['loss']
        total_loss['Reconstruction_Loss'] += loss['Reconstruction_Loss']
        total_loss['KLD'] += loss['KLD']

        optimizer.zero_grad()

        loss['loss'].backward()

        optimizer.step()

    total_loss['loss'] /= len(dataloader)
    total_loss['Reconstruction_Loss'] /= len(dataloader)
    total_loss['KLD'] /= len(dataloader)

    return total_loss


def sample_step(model: BaseVAE,
                epoch: int,
                num_samples: int,
                device: torch.device,
                writer: SummaryWriter):
    samples = model.sample(num_samples=num_samples,
                           current_device=device)

    image_grid = make_grid(samples)
    writer.add_image(f'#{epoch} Generated Samples', image_grid)


def reconstruction_step(model: BaseVAE,
                        epoch: int,
                        x: torch.tensor,
                        device: torch.device,
                        writer: SummaryWriter):
    x = x.to(device)
    result = model(x)

    image_grid_rc = make_grid(result[0])

    writer.add_image(f'#{epoch} Reconstructed images', image_grid_rc)


def train(model: BaseVAE,
          dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          epochs: int,
          writer: SummaryWriter,
          device: torch.device):

    x = next(iter(dataloader))[0][0:4]
    x = x.to(device)
    image_grid_gt = make_grid(x)
    writer.add_image('A Original_image', image_grid_gt)

    for epoch in tqdm(range(epochs)):
        loss = train_step(model,
                          dataloader,
                          optimizer,
                          device)
        scheduler.step()

        writer.add_scalar('training_loss',
                          loss['loss'],
                          epoch)

        writer.add_scalar('reconstruction_loss',
                          loss['Reconstruction_Loss'],
                          epoch)

        writer.add_scalar('KLD',
                          loss['KLD'],
                          epoch)

        sample_step(model,
                    epoch,
                    num_samples=4,
                    device=device,
                    writer=writer)

        reconstruction_step(model,
                            epoch,
                            x,
                            device=device,
                            writer=writer)

    writer.close()
