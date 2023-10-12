import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import accuracy_fn, plot_decision_boundary, plot_predictions


# ======================================================================================================
#                                         Multi-classification
'''
이전에는 Binary Classification이였다면, 이번에는 Multi-class
'''
# -----HYPER PARAMETERS: Dataset----#
NUM_CLASSES = 6
NUM_FEATURES = 2
RANDOM_SEED = 10
# ----------------------------------#
# ======================================================================================================


class BlobClassifier(nn.Module):
    def __init__(self,
                 input_features: int,
                 output_features: int,
                 hidden_units: int = 8,
                 only_linear: bool = False) -> None:
        '''
        Initializes multi-class classification model.

        Args:
            input_features (int): Number of input features to the model
            output_features (int): Number of output features (Number of output classes)
            hidden_units (int): Number of hidden units between layers, default 8
            only_linear (bool): if ```True```, Not use Non-Linear activation Function

        Returns:
            None

        Example:
            ```
            model = BlobClassifier(input_features=3,
                                   ouput_features=4,
                                   hidden_units=8)
            ```
        '''
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


# ======================================================================================================
#                                           Plot Dataset

def plot_scatter_datasets(X: torch.Tensor, y: torch.Tensor) -> None:
    '''
    Plot Make_Blokbs datatsets,
    Vizualize.
    '''
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()

# ======================================================================================================

# ======================================================================================================
#                                                Train


def train(model, X_train, y_train, loss_fn, optimizer):
    '''
    Convert models outputs(i.e. logits) to prediction probabilities and train the model. 
    Args:
        model (BlobClf)
        X_train
    '''
    model.train()

    # Output of model -> logits
    y_logits = model(X_train)

    # Probabilities -> Using Softmax -> Predicted Labels using argMax
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, acc

# ======================================================================================================

# ======================================================================================================
#                                                Eval


def test(model, X_test, y_test, loss_fn, optimizer):
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_test)
        y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss = loss_fn(y_logits, y_test)
        acc = accuracy_fn(y_true=y_test,
                          y_pred=y_preds)

    return loss, acc

# ======================================================================================================

# ======================================================================================================
#                                       Making Predictions


def plot_decision_boundary_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_test)
        y_probabilities = torch.softmax(y_logits, dim=1)
        y_pred_labels = torch.argmax(y_probabilities, dim=1)

    print(torch.eq(y_pred_labels[:20], y_test[:20]))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Train')
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title('Test')
    plot_decision_boundary(model, X_test, y_test)
    plt.show()


# ======================================================================================================


# ======================================================================================================
#                                                Main
if __name__ == '__main__':

    # Reproducibility
    torch.manual_seed(42)

    # 1. Create multi-class Data
    X, y = make_blobs(n_samples=1000,
                      n_features=NUM_FEATURES,
                      centers=NUM_CLASSES,
                      cluster_std=2,
                      random_state=RANDOM_SEED)

    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.LongTensor)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=RANDOM_SEED)

    # plot Dataset
    plot_scatter_datasets(X, y)

    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(
        device), y_train.to(device), y_test.to(device)

    # Create Model
    model = BlobClassifier(input_features=2,
                           output_features=NUM_CLASSES,
                           #    only_linear=True,
                           hidden_units=8).to(device)

    # Loss Functions & Optimizer
    # ----------------------------------------
    #           CrossEntropyLoss
    '''
    This Criterion computes the Cross Entropy Loss Between input and target.

    It is Useful when training a classification problem with C classes(i.e. multi-classes).
    If provided, the optional argument *weight* should be a 1D Tensor assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    NOTE: nn.CrossEntropyLoss는 내부적으로 Softmax함수를 적용하기 때문에, 수동으로 적용할 필요가 없음.
          logit을 입력값으로 기대함.
    '''
    # ----------------------------------------

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-1)

    # Train, Eval Loop
    epochs = 100
    for epoch in range(epochs):
        train_loss, train_acc = train(model=model,
                                      X_train=X_train,
                                      y_train=y_train,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer)
        test_loss, test_acc = test(model=model,
                                   X_test=X_test,
                                   y_test=y_test,
                                   loss_fn=loss_fn,
                                   optimizer=optimizer)

        if epoch % 10 == 0:
            print(
                f'Epoch: {epoch}\t|\tTrain Loss: {train_loss:.4f}\tTrain Acc: {train_acc:.2f}%\t|\tTest Loss: {test_loss:.4f}\tTest Acc: {test_acc:.2f}%')

    plot_decision_boundary_and_evaluate(model=model,
                                        X_test=X_test,
                                        X_train=X_train,
                                        y_test=y_test,
                                        y_train=y_train)


# ======================================================================================================
