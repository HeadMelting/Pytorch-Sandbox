import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# MODEL


class SimpleMLP(nn.Module):
    def __init__(self,
                 input_features: int,
                 output_features: int,
                 hidden_units: int = 8,
                 only_linear: bool = False) -> None:
        super().__init__()

        # Layers
        if only_linear:
            self.linear_layer_stack = nn.Sequential(
                nn.Linear(in_features=input_features,
                          out_features=hidden_units),
                nn.Linear(in_features=hidden_units,
                          out_features=hidden_units),
                nn.Linear(in_features=hidden_units,
                          out_features=output_features)
            )
        else:
            self.linear_layer_stack = nn.Sequential(
                nn.Linear(in_features=input_features,
                          out_features=hidden_units),
                nn.ReLU(),
                nn.Linear(in_features=hidden_units,
                          out_features=hidden_units),
                nn.ReLU(),
                nn.Linear(in_features=hidden_units,
                          out_features=hidden_units),
                nn.ReLU(),
                nn.Linear(in_features=hidden_units,
                          out_features=output_features)
            )

    def forward(self, x):
        return self.linear_layer_stack(x)


# Dataset
x, y = np.meshgrid(np.linspace(0, 1, 130), np.linspace(0, 1, 130))

features = torch.from_numpy(np.column_stack((x.ravel(), y.ravel())))
features = features.to(dtype=torch.float32)

# z1 = x + y
z1 = z1 = features[:, 0] + features[:, 1]

# z2 = (x-0.5)*(y-0.5)
z2 = (features[:, 0]-0.5) * (features[:, 1]-0.5)

# z3 = cos(x) * sin(y)
z3 = (features[:, 0]*10).cos() + (features[:, 1]*10).sin()


# device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

features = features.to(device)
z1, z2, z3 = z1.to(device), z2.to(device), z3.to(device)

# Init Models
model1 = SimpleMLP(input_features=2,
                   output_features=1,
                   hidden_units=64).to(device)
model2 = SimpleMLP(input_features=2,
                   output_features=1,
                   hidden_units=64).to(device)
model3 = SimpleMLP(input_features=2,
                   output_features=1,
                   hidden_units=64).to(device)


# Init optimizer and loss functions
loss_fn = nn.MSELoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-1)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-1)
optimizer3 = torch.optim.SGD(model3.parameters(), lr=1e-1)

# Train Test Split
X_train = features[:11000]
z1_train = z1[:11000].unsqueeze(1)
z2_train = z2[:11000].unsqueeze(1)
z3_train = z3[:11000].unsqueeze(1)

X_test = features[11000:]
z1_test = z1[11000:].unsqueeze(1)
z2_test = z2[11000:].unsqueeze(1)
z3_test = z3[11000:].unsqueeze(1)


# Train function
def train_epoch(epoch, model, loss_fn, optimizer, X_train, X_test, y_train, y_test):
    model.train()
    logit = model(X_train)
    loss = loss_fn(logit, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    test_logit = model(X_test)
    test_loss = loss_fn(test_logit, y_test)
    if epoch % 100 == 0:
        print(f'Epoch:{epoch} | TestLoss: {test_loss} | TrainLoss: {loss}')


def train():
    epochs = 5000
    print('Training Model1 | z = x + y')
    for epoch in range(epochs):
        train_epoch(epoch=epoch, model=model1,
                    loss_fn=loss_fn,
                    optimizer=optimizer1,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=z1_train,
                    y_test=z1_test)

    print('Training Model2 | z = x * y')
    for epoch in range(epochs):
        train_epoch(epoch=epoch, model=model2,
                    loss_fn=loss_fn,
                    optimizer=optimizer2,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=z2_train,
                    y_test=z2_test)

    print('Training Model3 | z = cos(x) + cos(y)')
    for epoch in range(epochs):
        train_epoch(epoch=epoch, model=model3,
                    loss_fn=loss_fn,
                    optimizer=optimizer3,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=z3_train,
                    y_test=z3_test)


def plot3d(X, X_train, X_test, y, model, title):
    y_train = model(X_train)
    y_test = model(X_test)

    X, X_train, X_test = X.cpu().numpy(), X_train.cpu().numpy(), X_test.cpu().numpy()
    y, y_train, y_test = y.detach().cpu().numpy(), y_train.detach(
    ).cpu().numpy(), y_test.detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[:, 0], X[:, 1], y, label='true')
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, label='train')
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label='test')
    fig.legend(fontsize='20')
    plt.title(title, fontsize='30')
    plt.show()


# train
train()

# plot model 1
plot3d(features, X_train, X_test, z1, model1,
       'z = x + y | 3000 Epoch')
# plot model 2
plot3d(features, X_train, X_test, z2, model2,
       'z = x * y | 3000 Epoch')
# plot model 3
plot3d(features, X_train, X_test, z3, model3,
       'z = cos(x) + sin(y)  | 3000 Epoch')
