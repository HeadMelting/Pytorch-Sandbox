from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


def get_train_val_dataloader(data_path, batch_size=32, shuffle=True, num_workers=2, **kwargs):

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.CenterCrop(148),
                                           transforms.Resize(64),
                                           transforms.ToTensor()])

    val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.CenterCrop(148),
                                         transforms.Resize(64),
                                         transforms.ToTensor()])

    train_dataset = MyCelebA(data_path,
                             split='train',
                             transform=train_transforms,
                             download=False)

    val_dataset = MyCelebA(data_path,
                           split='test',
                           transform=val_transforms,
                           download=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader
