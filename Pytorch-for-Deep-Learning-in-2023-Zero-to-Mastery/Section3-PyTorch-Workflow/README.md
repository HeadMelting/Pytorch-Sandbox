# PyTorch Workflow

1. Get data Ready

- turn into tensors

* torchvision.transforms
* torch.utils.data.Dataset
* torch.utils.data.DataLoader
  <br/>
  <br/>

2. Build or pick a pretrained model(to suit your problem)

- Pick a loss function & optimizer
- Build a training loop

* torch.nn
* torch.nn.Module
* torchvision.models
  <br/><br/>

3. Fit the model to the data and make a prediction
   <br/><br/>

4. Evaluate the model

- torchmetrics
  <br/><br/>

5. Improve through experimentation

- tensorboard
  <br/><br/>

6. Save and reload your trained model
