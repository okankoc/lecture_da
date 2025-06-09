"""Utility functions used for the Domain Adaptation code."""
import time
import torch
import numpy as np


def train(dataloader, model, loss_fun, optimizer, num_epochs, device, params=None, report_every=1, report_acc=True):
    size = len(dataloader.dataset)
    t0 = time.perf_counter()
    model.train()
    for epoch in range(num_epochs):
        # print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), one_hot(y, model.num_classes).to(device)
            # TODO: Is this necessary?
            X = X.contiguous()
            loss, params = step(optimizer, model, params, loss_fun, X, y)
            if batch % report_every == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} epoch:{epoch+1} [{current:>5d}/{size:>5d}]")
        if report_acc is True:
            print("Train dataset metrics:")
            test(dataloader, model, loss_fun, device)


def test(dataloader, model, loss_fun, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fun(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


# Convenient wrapper for stepping with first order AND custom second order methods
def step(optimizer, model, params, loss_fun, X, y):
    opt = optimizer["method"]
    # TODO: Check if optimizer is custom!
    if optimizer["name"] == "SGD" or optimizer["name"] == "ADAM":
        loss = loss_fun(model(X), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss, None
    else:
        params, loss = opt.step(model, params, loss_fun, X, y)
        return loss, params


def one_hot(x, k):
    return torch.nn.functional.one_hot(x, k).float()
