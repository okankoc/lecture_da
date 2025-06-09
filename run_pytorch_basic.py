# Copyright 2025 OKAN KOC

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Script for testing Empirical Risk Minimization using pytorch."""
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt

from models import ConvNet, MLP

matplotlib.use(backend="QtAgg", force=True)

# Get cpu, gpu or mps device for training.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def get_data():
    training_data = datasets.MNIST(
        root="data/",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root="data/",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data, test_data


def viz_data(data):
    figure = plt.figure(figsize=(3, 3))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        img = img.squeeze()
        if len(img.shape) == 3:
            img = img.permute([1, 2, 0])
        plt.imshow(img)
    plt.show()


def get_data_loader(training_data, test_data, batch_size):
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def eval_model(model, test_data):
    model.eval()
    x, y = test_data[0][0].unsqueeze(dim=0), test_data[0][1]
    with torch.no_grad():
        x = x.to(DEVICE)
        pred = model(x)
        predicted, actual = pred[0].argmax(0), y
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def run_network():
    # Get MNIST data
    train_data, test_data = get_data()
    # Visualize data to see what kind of data it is
    # viz_data(train_data)
    # Create data loader to feed data batchwise in the optimization
    train_dataloader, test_dataloader = get_data_loader(
        train_data, test_data, batch_size=128
    )

    # Create a model for the classifier
    model = ConvNet().to(DEVICE)
    # Try another model if you like
    # model = MLP().to(DEVICE)

    # Create a loss function (softmax + cross_entropy)
    loss_fn = nn.CrossEntropyLoss()

    # Create optimizer - Adam works really well for a lot of datasets/models.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Start the training procedure
    epochs = 2
    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        # Optional: Test throughout to see if model is learning well!
        test(test_dataloader, model, loss_fn)

    # Now the model is ready to use for inference
    # Optional: evaluate model on example test data
    eval_model(model, test_data)


if __name__ == "__main__":
    # Note that code is somewhat based on the Pytorch tutorials here: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    run_network()
