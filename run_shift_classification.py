# Copyright 2025 OKAN KOC

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Script for testing distribution shift adaptation with MNIST dataset."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib
import matplotlib.pyplot as plt

import utils
from adapt import DANN, IW, PseudoLabel, JDOT
from models import ConvNet, ConvNet2, ConvDomainClassifier

# Necessary in mac osx to be able close figures in emacs
matplotlib.use(backend="QtAgg", force=True)


def report_acc(scenario, model, loss_fun, report_train=False):
    if report_train:
        # These are very slow
        print(
            f"Reporting accuracy/loss on source {scenario.source_name} training dataset..."
        )
        utils.test(scenario.source_dataloader, model, loss_fun, scenario.device)

        print(
            f"Reporting accuracy/loss on target {scenario.target_name} training dataset..."
        )
        utils.test(scenario.target_dataloader, model, loss_fun, scenario.device)

    print(f"Reporting accuracy/loss on {scenario.source_name} test dataset...")
    utils.test(scenario.source_test_dataloader, model, loss_fun, device=scenario.device)

    print(f"Reporting accuracy/loss on {scenario.target_name} test dataset...")
    utils.test(scenario.target_test_dataloader, model, loss_fun, device=scenario.device)


def report_models_acc(scenario, methods, loss_fun, batch_idx):
    for i, method in enumerate(methods):
        print("===============================")
        print(f"Method {method.name}")
        report_acc(scenario, method.model, loss_fun)
        print("===============================")


def train_network(model, loss_fun, train_data, device, dataloader_options, num_epochs):
    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    # sgd = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0)
    opt = {
        "name": "ADAM",
        "method": adam,
        "num_epochs": num_epochs,
        "batch_size": dataloader_options["batch_size"],
    }
    params = dict(model.named_parameters())  # We actually don't need it for SGD
    train_dataloader = DataLoader(train_data, **dataloader_options)
    utils.train(
        train_dataloader,
        model,
        loss_fun,
        opt,
        num_epochs,
        device=device,
        params=params,
        report_every=10,
    )
    # Report accuracy/loss on whole training dataset
    utils.test(train_dataloader, model, loss_fun, device)


class MNIST_to_MNIST_M:
    def __init__(
        self, dataloader_options, gen_acc_curve, device, preprocess, conv_name
    ):
        if preprocess is True:
            self.process_MNIST_M_labels(root="data/MNIST-M", use_train=True)
            self.process_MNIST_M_labels(root="data/MNIST-M", use_train=False)
        self.dataloader_options = dataloader_options
        self.device = device
        self.source_name = "MNIST"
        self.target_name = "MNIST-M"
        self.conv_name = conv_name
        self.transforms_source = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),
                self.StackTransform(),
            ]
        )
        self.transforms_target = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.source_data = datasets.MNIST(
            root="data", train=True, download=True, transform=self.transforms_source
        )
        self.target_data = datasets.ImageFolder(
            root="data/MNIST-M/train", transform=self.transforms_target
        )

        # Load both datasets
        self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
        self.target_dataloader = DataLoader(self.target_data, **dataloader_options)

        source_test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=self.transforms_source
        )
        self.source_test_dataloader = DataLoader(source_test_data, **dataloader_options)
        target_test_data = datasets.ImageFolder(
            root="data/MNIST-M/test", transform=self.transforms_target
        )
        self.target_test_dataloader = DataLoader(target_test_data, **dataloader_options)

    def process_MNIST_M_labels(self, root, use_train):
        if use_train:
            folder = "train"
        else:
            folder = "test"
        labels_file = os.path.join(root, folder + "_labels.txt")
        out = pandas.read_csv(
            labels_file, header=None, names=["name", "label"], sep=" "
        )
        for label in out["label"].unique():
            os.makedirs(os.path.join(root, folder, str(label)), exist_ok=True)
        for index, elem in out.iterrows():
            os.rename(
                src=os.path.join(root, folder, elem["name"]),
                dst=os.path.join(root, folder, str(elem["label"]), elem["name"]),
            )

    class StackTransform:
        # Transform MNIST to SVHN RGB-shape (this is a bit dumb)
        def __call__(self, x):
            return torch.stack((x[0], x[0], x[0]), dim=0)

    def train_model(self, num_epochs):
        loss_fun = nn.CrossEntropyLoss()
        if self.conv_name == "conv":
            model = ConvNet(num_classes=10).to(self.device)
        elif self.conv_name == "conv2":
            model = ConvNet2(num_classes=10).to(self.device)
        else:
            raise Exception("Unknown model!")

        train_network(
            model,
            loss_fun,
            self.source_data,
            self.device,
            self.dataloader_options,
            num_epochs,
        )
        return model, loss_fun


def run_shift(scenario, num_epochs):
    gen_acc_curve = False
    model, loss_fun = scenario.train_model(num_epochs=2)

    # Prepare adaptation methods
    methods = []
    discriminator = ConvDomainClassifier()
    methods.append(
        DANN(
            model,
            loss_fun,
            discriminator=discriminator.to(scenario.device),
            layer_to_apply_disc="flatten",
            device=scenario.device,
            learning_rate=1e-3,
            num_epochs=num_epochs,
            num_batches=min(
                len(scenario.source_dataloader), len(scenario.target_dataloader)
            ),
        )
    )
    methods.append(IW(model, scenario.device, method="kmm", use_embedding=False))
    methods.append(PseudoLabel(model, loss_fun, learning_rate=1e-3, device=scenario.device, pl_type='hard', temp=1))
    methods.append(JDOT(model, loss_fun, scenario.device, alpha=1e-3, lamb=1e-3, learning_rate=1e-3))

    # Report accuracy of trained network on both datasets
    report_acc(scenario, model, loss_fun)

    batch_idx = 0
    # Run adaptation
    for k in range(num_epochs):
        print(f"Epoch {k+1}")
        for (X_train, y_train), (X_shift, y_shift) in zip(
            scenario.source_dataloader, scenario.target_dataloader
        ):
            X_train, X_shift, y_train = (
                X_train.to(scenario.device),
                X_shift.to(scenario.device),
                y_train.to(scenario.device),
            )
            for method in methods:
                method.adapt(X_train, y_train, X_shift)

            print(f"Batch id: {batch_idx+1}")
            batch_idx += 1

        report_models_acc(scenario, methods, loss_fun, batch_idx)


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dataloader_options = {"batch_size": 500, "shuffle": True, "drop_last": True}
    scenario = MNIST_to_MNIST_M(
        dataloader_options,
        gen_acc_curve=False,
        device=device,
        preprocess=False,
        conv_name="conv",
    )
    run_shift(scenario, num_epochs=5)
    plt.show()
