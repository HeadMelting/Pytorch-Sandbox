# Section 6: PyTorch Custom Datasets

알고 있는 내용들은 빠르게 넘어가기 위해, 모든 강의 내용을 정리하지는 않았음.

배운 내용은 아래에 간략하게 정리함.

1. `PyTorch`는 영상 Tensor 구조가, **Color_channels first**이지만, `matplotlib`이나 `PIL`은 **Color_channels last** 이므로, `torch.Tensor.permute()`을 이용하여 알맞게 변환해주어야함.

2. Input Image의 Width, Height는 제각각일 수 있음. 따라서, 모델에 알맞은 형태로 변환하기 위해 `resize`를 사용하여 `64 x 64`형태로 변환해주었음.

## ImageFolder

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

## DataLoader

```python
from torch.utils.data import DataLoader
import os
train_dataloader = DataLoader(dataset=train_Data,
                              batch_size = 1,
                              num_workers=os.cpu_count())
                              # num_workers = os.cpu_count() : 사용할 수 있는 모든 cpu core를 사용하게 됨.

```

- `num_workers`: How many CPU cores that is used to load data.
- `os.cpu_count()`: 얼마나 많은 cpu core를 가지고 있는지 알 수 있음.
