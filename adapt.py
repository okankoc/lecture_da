# Domain adaptation approaches
import copy
import torch
from torch import nn
from torch.autograd import Function
import numpy as np
import ot

import utils
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
            w = estimate_gauss_ratio(x_shift, x_train)
        else:
            raise Exception('Unknown ratio estimation method!')
        # Update SGD with those ratios
        loss = torch.sum(self.loss_fun(self.model(X_train), y_train) * w.to(self.device))
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()


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


# Code adapted from: https://github.com/fungtion/DANN
class DANN:
    def __init__(
        self, model, loss_fun, discriminator, layer_to_apply_disc, device, learning_rate, num_epochs, num_batches
    ):
        self.name = "DANN"
        self.device = device
        self.model = model.copy(device)
        self.model.track_features(layer_to_apply_disc)
        self.discriminator = discriminator.copy(device)
        self.opt_model = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        self.loss_class = copy.deepcopy(loss_fun).to(device)
        self.loss_domain = copy.deepcopy(loss_fun).to(device)
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.idx = 0
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.discriminator.parameters():
            p.requires_grad = True

    def forward_adversarial(self, X_data):
        epoch_idx = self.idx // self.num_batches
        p = (self.idx + epoch_idx * self.num_batches) / (self.num_epochs * self.num_batches)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        input_data = torch.as_tensor(X_data, dtype=torch.float32)
        class_output = self.model(input_data)
        # TODO: This gradient reversal seems very unnecessary if we maintain two optimizers
        reverse_feature = ReverseLayerF.apply(self.model.features, alpha)
        domain_output = self.discriminator(reverse_feature)
        return class_output, domain_output

    def adapt(self, X_source, y_source, X_target, y_target=[]):
        source_batch_size = X_source.shape[0]

        # Feeding in source inputs
        domain_label = utils.one_hot(torch.zeros(source_batch_size, device=self.device, dtype=torch.long), 2)
        class_output, domain_output = self.forward_adversarial(X_source)
        err_s_label = self.loss_class(class_output, y_source)
        err_s_domain = self.loss_domain(domain_output, domain_label)

        # Feeding in target labels
        target_batch_size = X_target.shape[0]
        domain_label = utils.one_hot(torch.ones(target_batch_size, device=self.device, dtype=torch.long), 2)
        _, domain_output = self.forward_adversarial(X_target)

        err_t_domain = self.loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        self.opt_model.step()
        self.opt_model.zero_grad()
        self.opt_disc.step()
        self.opt_disc.zero_grad()
        self.idx += 1


class PseudoLabel:
    def __init__(self, model, loss_fun, learning_rate, device, pl_type='hard', temp=10):
        self.model = model.copy(device)
        self.loss_fun = copy.deepcopy(loss_fun)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        # For debugging
        self.name = "PseudoLabel"
        self.pl_type = pl_type
        self.temp = temp # in case we use soft pseudo-labeling

    # Pseudo-label based adaptation
    def adapt(self, X_train, y_train, X_shift):
        loss = self.loss_fun(self.model(X_train), y_train)
        y_shift_pred = self.model(X_shift)
        if self.pl_type == 'hard':
            y_shift_pseudo = torch.argmax(y_shift_pred, dim=1)
            y_shift_pseudo = utils.one_hot(y_shift_pseudo, self.model.num_classes)
        else:
            y_shift_pseudo = torch.softmax(y_shift_pred * self.temp, dim=1)
        loss += self.loss_fun(self.model(X_shift), y_shift_pseudo)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()


class JDOT:
    def __init__(
        self,
        model,
        loss_fun,
        device,
        alpha,
        lamb,
        learning_rate,
        num_iter=1,
        use_layer="last_features",
        add_source_loss=True,
        use_squared_dist=False,
    ):
        self.model = model.copy(device)
        self.loss_fun = copy.deepcopy(loss_fun)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        self.name = "JDOT"
        self.debug_loss = []
        self.debug_acc = []
        self.alpha = alpha
        self.lamb = lamb
        self.num_iter = num_iter
        self.model.track_features(use_layer)
        self.add_source_loss = add_source_loss
        self.use_squared_dist = use_squared_dist
        self.device = device

    def adapt(self, X_train, y_train, X_shift, y_shift=[]):
        num_target = X_shift.shape[0]
        results = {
            "acc": torch.zeros(self.num_iter),
            "loss": torch.zeros(self.num_iter),
            "w_dist": torch.zeros(self.num_iter),
        }
        for k in range(self.num_iter):
            # print(f"JDOT Iter: {k}")
            prob, w_dist = self.transport(X_train, X_shift, y_train)
            # Debugging
            if len(y_shift) != 0:
                y_pred = self.model(X_shift)
                results["loss"][k] = self.loss_fun(y_pred, y_shift)
                results["acc"][k] = torch.mean((y_pred.argmax(1) == y_shift.argmax(1)).type(torch.float))
                results["w_dist"][k] = w_dist
            loss = self.loss(X_train, y_train, X_shift, prob)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return results

    def transport(self, X_source, X_target, y_source):
        num_source = X_source.shape[0]
        num_target = X_target.shape[0]
        # Weights of the points
        w_source = torch.ones(num_source) / num_source
        w_target = torch.ones(num_target) / num_target

        self.model(X_source)
        source_activations = torch.clone(self.model.features)
        pred_target = self.model(X_target)
        target_activations = torch.clone(self.model.features)
        cost_mat = ot.utils.euclidean_distances(source_activations, target_activations, squared=self.use_squared_dist)
        cost_mat = self.alpha * cost_mat + self.lamb * self.calc_loss_mat(y_source, pred_target)
        prob_mat = ot.emd(a=w_source, b=w_target, M=cost_mat).type(torch.float).to(self.device)
        if self.use_squared_dist is True:
            return prob_mat, torch.sqrt(torch.sum(prob_mat * cost_mat))
        return prob_mat, torch.sum(prob_mat * cost_mat)

    def calc_loss_mat(self, y_source, y_pred):
        num_source = y_source.shape[0]
        num_target = y_pred.shape[0]
        ys = torch.repeat_interleave(y_source, num_target, dim=0)
        yt = y_pred.repeat(num_source, 1)
        # FIXME: This won't work if loss_fun is different from cross-entropy!!!
        loss_fun = nn.CrossEntropyLoss(reduction="none")
        return loss_fun(yt, ys).reshape(num_source, num_target)

    def loss(self, X_train, y_train, X_shift, prob_mat):
        source_loss = 0.0
        val = self.loss_fun(self.model(X_train), y_train)
        if self.add_source_loss is True:
            source_loss += val
        source_activations = torch.clone(self.model.features)

        idx_source, idx_target = torch.where(prob_mat)
        probs = prob_mat[torch.where(prob_mat)]
        y_pred = self.model(X_shift)
        layer_loss = torch.sum(probs * torch.dist(self.model.features[idx_target], source_activations[idx_source]))
        self.loss_fun.reduction = "none"
        losses = self.loss_fun(y_pred[idx_target], y_train[idx_source])
        self.loss_fun.reduction = "mean"
        if len(losses.shape) == 2:
            losses = torch.mean(losses, dim=1)
        target_loss = torch.sum(probs * losses)
        return source_loss + self.alpha * layer_loss + self.lamb * target_loss
