# PyTorch Computer Vision

**Udemy Section5**

## What is a Computer Vision?

- Image Classification
  - is this photo of sushi, steack or pizza?

* Object detection
  - Where's the car we're looking for?
  - Position of object on image

- Image Segmentation
  - What are the different sections in this image?
  - [MachineLearning.apple.com](https://machinelearning.apple.com/research/panoptic-segmentation)

* Tesla Computer Vision
  - Self driving cars
  - [Tesla AI Day 49:49](https://youtube.com/watch?t=2989&v=j0z4FweCy4M&feature=youtu.be)
  - [Tesla AI Day 2:01:31](https://youtube.com/watch?v=j0z4FweCy4M&t=7291s)

## What we're going to cover (broadly)

- Getting a vision dataset to work with using torchvision.datasets
- Architecture of a CNN with PyTorch
- An end-to-end multi-class image classification problem
- Steps in modelling with CNNs in PyTorch
  - Creating a CNN model with PyTorch
  - Picking a loss and optimizer
  - Training a model
  - Evaluating a model

## Computer Vision inputs and outputs

- input: RGB(3channel) image. | shape of (C, H, W)

  - ex) [3, 224, 224]

- Image -> Numerical encoding(normalized pixel values) -> model(CNN) -> outputs

- PyTorch: Image Shape should be **NCHW(Batch_size, Color_channels, Height, Width)**

## Flow

1. Get data Ready (turn into tensors)

- torch.vision.transforms
- torch.utils.data.Dataset
- torch.utils.data.DataLoader
  <br/><br/>

2. Build or pick a pretrained model

- loss functions & optimizer
- Build a training loop

* torch.optim
* torch.nn
* torch.nn.Module
* torchvision.models
  <br/><br/>

3. Fit the model to the data and make a prediction
   <br/><br/>

4. Evaluate the model

- torchmetrics
  <br/><br/>

5. Improvce through experimentation

- torch.utils.tensorboard
  <br/><br/>

6. Save and reload your trained model

- torch.save
- torch.load
  <br/><br/>

# CNN

| Hyperparameter/LayerType                | What does it do?                                                  | Typical values                                                           |
| --------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Input image(s)                          | Target images you'd like to discover patterns in                  | Whatever you can take a photo (or video) of                              |
| Input layer                             | Takes in target images and preprocesses them for further layers   | PyTorch image shape should be [batch_size, color channel, Height, Width] |
| Convolution layer                       | Extracts/learns the most important features from target images    | Multiple, can create with `torch.nn.ConvXd()`                            |
| Hidden activation/non-linear activation | Adds non-linearity to learned features                            | Usually ReLU (`torch.nn.ReLU()`), though can be many more                |
| Pooling layer                           | Reduces the dimensionality of learned image features              | Max (`torch.nn.Maxpool2d()`) or Average (`torch.nn.AvgPool2d()`)         |
| Ouput layer/linear layer                | Takes learned features and outputs them in shape of target labels | `torch.nn.Linear(out_features = [num_of_class])`                         |
| Output activation                       | Converts output logits to prediction probabilities                | `torch.sigmoid()` for binary or `torch.softmax` for mutli-class          |

## Sample Code

```python
import torch
from torch import nn
class CNN(nn.Module):
  def __init__(self,
              input_shape: int,
              hidden_units: int,
              output_shape: int):
    super().__init__()

    self.cnn_layers = nn.Sequential(
      nn.Conv2d(in_channels = input_shape,
                out_channels = hidden_units,
                kernel_size = 3, # how big is the square that's going over the image
                stride = 1, # take a step one pixel at a time
                padding = 1), # add an extra pixel around the input image
      nn.ReLU(), # non-linear activation layer
      nn.MaxPool2d(kernel_size = 2,
                    stride = 2)
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features = hidden_units * 32 * 32,
                out_features = output_shape)
    )

  def forward(self, x:torch.Tensor):
    x = self.cnn_layers(x)
    x = self.classifier(x)
    return x

cnn_model = CNN(input_shape = 3,
                hidden_units = 3,
                output_shape = 3)

```
