import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

# Used to optimize DANN (features are optimized to maximize domain classifier error).
# TODO: Is this necessary? Replace with 2 optimizers: one min. and one max.
# Alternatively, pytorch guide suggests using 'hooks'
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


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


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same")
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same")
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same")
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")
        self.dense = nn.LazyLinear(128)
        self.final = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = F.relu(x)
        return self.final(x)

# Taken from DANN code in https://github.com/fungtion/DANN
# TODO: Separate feature + classifier from domain classifier?
class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = 10

        self.feature = nn.Sequential()
        self.feature.add_module("f_conv1", nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module("f_bn1", nn.BatchNorm2d(64))
        self.feature.add_module("f_pool1", nn.MaxPool2d(2))
        self.feature.add_module("f_relu1", nn.ReLU(True))
        self.feature.add_module("f_conv2", nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module("f_bn2", nn.BatchNorm2d(50))
        self.feature.add_module("f_drop1", nn.Dropout2d())
        self.feature.add_module("f_pool2", nn.MaxPool2d(2))
        self.feature.add_module("f_relu2", nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module("c_fc1", nn.LazyLinear(100))
        self.class_classifier.add_module("c_bn1", nn.BatchNorm1d(100))
        self.class_classifier.add_module("c_relu1", nn.ReLU(True))
        self.class_classifier.add_module("c_drop1", nn.Dropout2d())
        self.class_classifier.add_module("c_fc2", nn.Linear(100, 100))
        self.class_classifier.add_module("c_bn2", nn.BatchNorm1d(100))
        self.class_classifier.add_module("c_relu2", nn.ReLU(True))
        self.class_classifier.add_module("c_fc3", nn.Linear(100, 10))
        self.class_classifier.add_module("c_softmax", nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module("d_fc1", nn.LazyLinear(100))
        self.domain_classifier.add_module("d_bn1", nn.BatchNorm1d(100))
        self.domain_classifier.add_module("d_relu1", nn.ReLU(True))
        self.domain_classifier.add_module("d_fc2", nn.Linear(100, 2))
        self.domain_classifier.add_module("d_softmax", nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        feature = self.feature(input_data)
        self.feature_out = nn.Flatten()(feature)
        return self.class_classifier(self.feature_out)

    def forward_adversarial(self, input_data, alpha):
        feature = self.feature(input_data)
        self.feature_out = nn.Flatten()(feature)
        reverse_feature = ReverseLayerF.apply(self.feature_out, alpha)
        class_output = self.class_classifier(self.feature_out)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    def copy(self, device):
        new_model = ConvNet2().to(device)
        new_model.load_state_dict(self.state_dict())
        new_model.save_params()
        return new_model

    # TODO: Remove this?
    def get_layer(self, layer_id):
        # For now we only return the pre-defined feature outputs!!!
        return self.feature_out

    @torch.no_grad()
    def save_params(self):
        self.state = copy.deepcopy(self.state_dict())

    @torch.no_grad()
    def restore_params(self):
        self.load_state_dict(self.state)
        return dict(self.named_parameters())
