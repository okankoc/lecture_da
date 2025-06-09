import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function


# Modified from DANN script
class ConvNet2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = []
        self.name = "conv2"

        self.net = nn.Sequential()
        self.net.add_module("conv1", nn.LazyConv2d(64, kernel_size=5))
        self.net.add_module("bn1", nn.BatchNorm2d(64))
        self.net.add_module("pool1", nn.MaxPool2d(2))
        self.net.add_module("relu1", nn.ReLU(True))
        self.net.add_module("conv2", nn.Conv2d(64, 50, kernel_size=5))
        self.net.add_module("bn2", nn.BatchNorm2d(50))
        self.net.add_module("drop1", nn.Dropout())
        self.net.add_module("pool2", nn.MaxPool2d(2))
        self.net.add_module("relu2", nn.ReLU(True))
        self.net.add_module("flatten", nn.Flatten())
        self.net.add_module("fc1", nn.LazyLinear(100))
        self.net.add_module("bn3", nn.BatchNorm1d(100))
        self.net.add_module("relu3", nn.ReLU(True))
        self.net.add_module("drop2", nn.Dropout())
        self.net.add_module("fc2", nn.Linear(100, 100))
        self.net.add_module("bn4", nn.BatchNorm1d(100))
        self.net.add_module("last_features", nn.ReLU(True))
        self.net.add_module("last_layer", nn.Linear(100, num_classes))

    def forward(self, input_data):
        return self.net(input_data)

    def copy(self, device):
        new_model = ConvNet2(self.num_classes).to(device)
        new_model.load_state_dict(self.state_dict())
        new_model.save_params()
        return new_model

    # Call model after this function to get layer outputs
    def track_features(self, layer_id):
        # Register hooks for the layers you're interested in
        def fun(module, inputs, outputs):
            self.features = outputs

        conv1_hook = self.net.get_submodule(layer_id).register_forward_hook(fun)

    @torch.no_grad()
    def save_params(self):
        self.state = copy.deepcopy(self.state_dict())

    @torch.no_grad()
    def restore_params(self):
        self.load_state_dict(self.state)
        return dict(self.named_parameters())


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.name = "conv1"
        self.features = []
        self.net = nn.Sequential()
        self.net.add_module("conv1", nn.LazyConv2d(32, kernel_size=3, stride=1, padding="same"))
        self.net.add_module("relu1", nn.ReLU(True))
        self.net.add_module("conv2", nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same"))
        self.net.add_module("relu2", nn.ReLU(True))
        self.net.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.net.add_module("conv3", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"))
        self.net.add_module("relu3", nn.ReLU(True))
        self.net.add_module("conv4", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"))
        self.net.add_module("relu4", nn.ReLU(True))
        self.net.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.net.add_module("conv5", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"))
        self.net.add_module("relu5", nn.ReLU(True))
        self.net.add_module("conv6", nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same"))
        self.net.add_module("relu6", nn.ReLU(True))
        self.net.add_module("pool3", nn.MaxPool2d(kernel_size=2, stride=2))
        self.net.add_module("flatten", nn.Flatten())
        self.net.add_module("fc1", nn.LazyLinear(128))
        self.net.add_module("last_features", nn.ReLU())
        self.net.add_module("last_layer", nn.Linear(128, num_classes))

    def copy(self, device):
        new_model = ConvNet(self.num_classes).to(device)
        new_model.load_state_dict(self.state_dict())
        new_model.save_params()
        return new_model

    # Call model after this function to get layer outputs
    def track_features(self, layer_id):
        # Register hooks for the layers you're interested in
        def fun(module, inputs, outputs):
            self.features = outputs

        conv1_hook = self.net.get_submodule(layer_id).register_forward_hook(fun)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def save_params(self):
        self.state = copy.deepcopy(self.state_dict())

    @torch.no_grad()
    def restore_params(self):
        self.load_state_dict(self.state)
        return dict(self.named_parameters())


# Separated domain classifier into a new class
class ConvDomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module("d_fc1", nn.LazyLinear(100))
        self.domain_classifier.add_module("d_bn1", nn.BatchNorm1d(100))
        self.domain_classifier.add_module("d_relu1", nn.ReLU(True))
        self.domain_classifier.add_module("d_fc2", nn.Linear(100, 2))
        self.domain_classifier.add_module("d_softmax", nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        return self.domain_classifier(input_data)

    def copy(self, device):
        new_model = ConvDomainClassifier().to(device)
        new_model.load_state_dict(self.state_dict())
        return new_model


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 200), nn.ReLU(), nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
