from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import transforms
import pandas as pd
import matplotlib.pyplot as plt

from os import path


class MyCelebA(Dataset):
    def __init__(self,
                 data_dir,
                 partition: int = -1,
                 image_transform=None,
                 target_transform=None,
                 label_file='identity_CelebA.txt',
                 image_dir='img_align_celeba',
                 partition_file='list_eval_partition.txt',
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.transform = image_transform
        self.target_transform = target_transform
        self.image_dir = image_dir

        dir_labels = path.join(data_dir, label_file)
        df_labels = pd.read_csv(dir_labels, sep=' ', header=None, names=[
                                'img_file', 'labels'])
        dir_partitions = path.join(data_dir, partition_file)
        df_partitions = pd.read_csv(dir_partitions, sep=' ', header=None, names=[
                                    'img_file', 'partition'])

        image_labels = pd.merge(df_labels, df_partitions, on='img_file')
        if partition != -1:
            image_labels = image_labels[image_labels['partition'] == partition]

        self.image_labels = image_labels

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        image_path = path.join(
            self.data_dir, self.image_dir, self.image_labels.iloc[index, 0])
        image = read_image(image_path)
        label = self.image_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_dataLoader(data_dir, partition: int = -1, batch_size=32, patch_size=64, transform=None, shuffle=True, num_workers=0, **kwargs):

    if not transform:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(patch_size),
                                        transforms.ToTensor(),])

    dataset = MyCelebA(data_dir,
                       image_transform=transform,
                       partition=partition,
                       **kwargs)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader


if __name__ == '__main__':
    train_dataloader = get_dataLoader('Data/celeba',
                                      partition=0,
                                      batch_size=1,
                                      shuffle=False)
    val_dataloader = get_dataLoader('Data/celeba',
                                    partition=1,
                                    batch_size=1,
                                    shuffle=False)
    test_dataloader = get_dataLoader('Data/celeba',
                                     partition=2,
                                     batch_size=1,
                                     shuffle=False)

    train_image, train_label = next(iter(train_dataloader))
    print(train_image.shape, train_label)
    val_image, val_label = next(iter(val_dataloader))
    print(val_image.shape, val_label)
    test_image, test_label = next(iter(test_dataloader))
    print(test_image.shape, test_label)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(train_image.squeeze().permute(1, 2, 0))
    plt.subplot(1, 3, 2)
    plt.imshow(val_image.squeeze().permute(1, 2, 0))
    plt.subplot(1, 3, 3)
    plt.imshow(test_image.squeeze().permute(1, 2, 0))
    plt.show()
