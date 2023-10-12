import torch
from torch import nn
import matplotlib.pyplot as plt
import os

# Plot function


def plot_predctions(X_train,
                    y_train,
                    X_test,
                    y_test,
                    predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(X_train, y_train, c='b', s=4, label='Training data')
    plt.scatter(X_test, y_test, c='r', s=4, label='Testing data')
    if predictions is not None:
        plt.scatter(X_test, predictions, c='g', s=4, label='Prediction')

    plt.legend(prop={'size': 14})
    plt.show()


# Model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()

        # Use nn.Linear
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

        # legacy
        '''
        self.weight = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))

         self.bais = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


if __name__ == '__main__':
    # Reproducibility
    torch.manual_seed(20)

    # Create Device-agnostic code.
    # if we've got access to GPU, our code will use it.
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using Device: {device}')

    # Data
    weight_gt = 4
    bias_gt = 2

    start = 0
    end = 1
    step = 0.02

    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = X * weight_gt + bias_gt

    # Data Split
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    print(f'Train Samples Count: {len(X_train)}')
    print(f'Test Samples Count: {len(X_test)}')

    plot_predctions(X_train, y_train, X_test, y_test)

    # Train Loop
    model = LinearRegression()
    model = model.to(device)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.2)

    epochs = 200

    # Data to GPU
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    for epoch in range(epochs):
        model.train()

        y_pred_train = model(X_train)
        loss = loss_fn(y_pred_train, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            y_pred_test = model(X_test)
            loss_test = loss_fn(y_pred_test, y_test)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch} | Loss: {loss} | Test Loss: {loss_test}')

    plot_predctions(X_train.cpu().numpy(), y_train.cpu().numpy(), X_test.cpu().numpy(), y_test.cpu().numpy(),
                    y_pred_test.cpu().numpy())

    # Save model
    MODEL_PATH = 'models'
    MODEL_NAME = 'putting_it_all_together_model.pth'

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    torch.save(model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME))

    # Load Model
    loaded_model = LinearRegression()
    loaded_model.to(device)
    loaded_model.load_state_dict(torch.load(
        os.path.join(MODEL_PATH, MODEL_NAME)))

    with torch.inference_mode():
        loaded_model_pred = loaded_model(X_test)
    plot_predctions(X_train.cpu().numpy(), y_train.cpu().numpy(), X_test.cpu().numpy(), y_test.cpu().numpy(),
                    loaded_model_pred.cpu().numpy())

    print('Loaded Model: ', loaded_model.state_dict())
    print('Original Model: ', model.state_dict())
