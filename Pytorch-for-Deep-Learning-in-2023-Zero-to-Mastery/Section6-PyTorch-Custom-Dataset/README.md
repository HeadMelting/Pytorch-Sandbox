# Section 6: PyTorch Custom Datasets

알고 있는 내용들은 빠르게 넘어가기 위해, 모든 강의 내용을 정리하지는 않았음.

배운 내용은 아래에 간략하게 정리함.

1. `PyTorch`는 영상 Tensor 구조가, **Color_channels first**이지만, `matplotlib`이나 `PIL`은 **Color_channels last** 이므로, `torch.Tensor.permute()`을 이용하여 알맞게 변환해주어야함.

2. Input Image의 Width, Height는 제각각일 수 있음. 따라서, 모델에 알맞은 형태로 변환하기 위해 `resize`를 사용하여 `64 x 64`형태로 변환해주었음.

<br/>

---

<br/>

## Loading Image Data wit `ImageFolder`

로컬에 저장된 모든 이미지를 `torchvision.datasets.ImageFolder`를 이용해서 Dataset으로 변환함.

```python
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform, # 데이터 변환
                                  target_transform=None)
```

### 이미지 폴더 구조

```bash
└── data(root)
    ├── train
    │   ├── pizza
    │   │   ├── 123412.png
    │   │   └── ...more images
    │   ├── steak
    │   ├── sushi
    │   └── ...more labels
    └── test
        ├── pizza
        │   ├── 22312.png
        │   └── ...more images
        ├── steak
        ├── sushi
        └── ...more labels
```

<br/>

---

<br/>

## DataLoader

```python
from torch.utils.data import DataLoader
import os
train_dataloader = DataLoader(dataset=train_Data,
                              batch_size = 32,
                              num_workers=os.cpu_count())
                              # num_workers = os.cpu_count() : 사용할 수 있는 모든 cpu core를 사용하게 됨.

img, label = next(iter(train_dataloader))
image.shape # [32, 3, 64, 64] | [batch_size, channels, H, W]

```

- `num_workers`: How many CPU cores that is used to load data.
- `os.cpu_count()`: 얼마나 많은 cpu core를 가지고 있는지 알 수 있음.

<br/>

---

<br/>

## Loading Image Data with Custom `Dataset`

1. want to be able to load images from file
2. Want to be able to get class names from the Dataset
3. Want to be able to get classes as dictionary from Dataset

Pros:

- Can create a `Dataset` out of almost anything
- Not limited to PyTorch pre-built `Dataset` functions

Cons:

- Even though you could create `Dataset` out of almost anything, it doesn't mean it will work.
- Using a custom `Dataset` often results in us writing more code, which could be prone to errors or performance issues.

```python
from torch.utils.data import Dataset
class ImageFolderCustom(Dataset):
   def __init__(self,
                tar_dir: str,
                transform = None):

       # Get all Images path
       self.paths = list(pathlib.Path(tar_dir).glob('*/*.jpg'))

       # Setup Transform
       self.transform = transform

       self.classes, self.class_to_idx = find_classes(tar_dir)

    def load_image(self, index: int):
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx

```

<br/>

---

<br/>

## Data Augmentation

Data augmentation is the process of artificially adding diversity to your training data.

### TrivialAugment

```python
from torchvision import transforms
train_transform = transforms.Compose([transforms.Resize(size=(224,224),
                                       transforms.TrivialAugmentWide(num_magnitude_bin=3),# 0 ~ 31
                                       transforms.ToTensor()
                                       ])

```

<br/>

---

<br/>

## Dealing with `over-fitting`

| <div style="width:max-content; text-align: center">Method to improve a model<br/>(reduce over-fitting)</div> | What does it do?                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Get more data                                                                                                | Gives a model more of a chance to learn patterns between samples(e.g. if a model is performing poorly on images of pizza, show it more images of pizza).                                                                                                                                           |
| Data augmentation                                                                                            | Increase the diversity of your training dataset without collecting more data(e.g. take your photos of pizza and randomly rotate). Increased diversity forces a model to learn more generalized patterns.                                                                                           |
| Better data                                                                                                  | Not all data samples are created equally. Removing poor samples from or adding better samples to dataset can improve model's performance.                                                                                                                                                          |
| Use transfer learning                                                                                        | Take a model's pre-learned patterns from one problem and tweak them to suit your own problem. (e.g. take a model trained on pictures of cars to recognize pictures of trucks).                                                                                                                     |
| Simplify a model                                                                                             | If the model is over-fitting, it may be too complicated model.                                                                                                                                                                                                                                     |
| Use learning rate decay                                                                                      | The idea here is to slowly decrease the learning rate as a model trains. This is akin to reaching for a coin at the back of a couch. The closer you get, the smaller your steps. The same with the learning rate, the closer you get to convergence, the smaller you'll want weight updates to be. |
| Use early stopping                                                                                           | Early stopping stops model training `before` it begins to over-fit.                                                                                                                                                                                                                                |

## Dealing with `under-fitting`

| <div style="width:max-content; text-align: center">Method to improve a model<br/>(reduce under-fitting)</div> | What does it do?                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Add more layers/units to your model                                                                           | if model is under-fitting, it may not have enough capability to `learn` the required patterns/weights/representations of the data to be predictive. |
| Tweak the learning rate                                                                                       | Perhaps your model's learning rate is too high to begin with. And it's updating weights too much, you might lower the lr                            |
| Train for longer                                                                                              | Sometimes a model just needs more time to learn representations of data.                                                                            |
| Use transfer learning                                                                                         | Take model's pre-learned patterns from one problem and tweak them to suit your own problem.                                                         |
| Use less regularization                                                                                       | Perhaps your model is under-fitting because you're trying to prevent over-fitting too much.                                                         |
