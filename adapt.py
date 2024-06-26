# Domain adaptation approaches
import torch
from torch import nn
import numpy as np

from ratio_estimation import kernel_mean_matching, estimate_gauss_ratio


# Importance weighting
class IW:
    class Logistic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.LazyLinear(out_features=2, bias=True)
        def forward(self, x):
            return self.layer(x)

    def __init__(self, model, device, method, use_embedding):
        self.device = device
        self.method = method
        self.use_embedding = use_embedding
        self.model = model.copy(device)
        self.loss_fun = nn.CrossEntropyLoss(reduction='none')
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        self.name = "Importance Weighting"
        self.ratio_model = self.Logistic().to(device)
        self.ratio_opt = torch.optim.Adam(self.ratio_model.parameters(), lr=2e-4)
        self.ratio_loss_fun = torch.nn.CrossEntropyLoss()
        self.num_ratio_batches = 100

    def estimate_ratio_logreg(self, X_shift, X_train):
        num_train = X_train.shape[0]
        num_shift = X_shift.shape[0]
        x_all = torch.vstack((X_train, X_shift))
        y_all = torch.hstack((torch.zeros(num_train), torch.ones(num_shift))).long().to(self.device)
        for batch in range(self.num_ratio_batches):
            loss = self.ratio_loss_fun(self.ratio_model(x_all), y_all)
            loss.backward()
            self.ratio_opt.step()
            self.ratio_opt.zero_grad()
            # print(f"loss: {loss:>7f} batch:{batch+1}")
        # print(f"Final classification error of ratio model: {loss:>7f}")
        probs = torch.nn.Softmax()(self.ratio_model(x_all[:num_train]))
        w_est = (num_train / num_shift) * probs[:, 1] / probs[:, 0]
        return w_est

    def adapt(self, X_train, y_train, X_shift):
        with torch.no_grad():
            if self.use_embedding:
                x_train = self.model(X_train)
                x_shift = self.model(X_shift)
            else:
                x_train = torch.nn.Flatten()(X_train)
                x_shift = torch.nn.Flatten()(X_shift)
        # Estimate ratio using e.g. Kernel Mean Embedding
        if self.method == 'logreg':
            w = self.estimate_ratio_logreg(x_shift, x_train)
        elif self.method == 'kmm':
            w = kernel_mean_matching(x_shift, x_train)
        elif self.method == 'de':
            # TODO: Implement a robust method to extract nonsingular dimensions for singular covariance matrices!
            w = estimate_gauss_ratio(x_shift, x_train)
        else:
            raise Exception('Unknown ratio estimation method!')
        # Update SGD with those ratios
        loss = torch.sum(self.loss_fun(self.model(X_train), y_train) * w)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()


# Code adapted from: https://github.com/fungtion/DANN
class DANN:
    # TODO: Models should be agnostic to DANN, but for now we expect model to define features and class, domain classifiers!
    def __init__(self, model, device, learning_rate, num_epochs, num_batches):
        self.name = "DANN"
        self.device = device
        self.model = model.copy(device)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        self.loss_class = torch.nn.NLLLoss().to(device)
        self.loss_domain = torch.nn.NLLLoss().to(device)
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.idx = 0
        for p in self.model.parameters():
            p.requires_grad = True

    def adapt(self, X_source, y_source, X_target, y_target=[]):
        batch_size = X_source.shape[0]
        epoch_idx = self.idx // self.num_batches
        p = (self.idx + epoch_idx * self.num_batches) / (self.num_epochs * self.num_batches)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        # Feeding in source inputs
        domain_label = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        class_output, domain_output = self.model.forward_adversarial(
            input_data=torch.as_tensor(X_source, dtype=torch.float32), alpha=alpha
        )
        err_s_label = self.loss_class(class_output, torch.as_tensor(y_source, dtype=torch.long))
        err_s_domain = self.loss_domain(domain_output, domain_label)

        # Feeding in target labels
        domain_label = torch.ones(batch_size, device=self.device, dtype=torch.long)
        _, domain_output = self.model.forward_adversarial(
            input_data=torch.as_tensor(X_target, dtype=torch.float32), alpha=alpha
        )

        err_t_domain = self.loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.idx += 1
