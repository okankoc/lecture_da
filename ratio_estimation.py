# Kernel mean matching
import torch
import numpy as np
from scipy.spatial.distance import cdist
import quadprog
import sklearn.covariance

# Learn the importance weighting assuming two distributions are gaussian
def estimate_gauss_ratio(x_test, x_train):
    cov_est = sklearn.covariance.LedoitWolf()
    p_gauss = lambda x, mu, S2: (1 / torch.sqrt(pow(2 * torch.pi, len(x)) * torch.det(S2))) * torch.exp(-0.5 * (x - mu) @ torch.linalg.lstsq(S2, x - mu)[0])
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    w = torch.zeros(num_train)
    mu_test = torch.sum(x_test, dim=0) / num_test
    S2_test = torch.tensor(cov_est.fit(x_test - mu_test).covariance_).float()
    mu_train = torch.sum(x_train, dim=0) / num_train
    S2_train = torch.tensor(cov_est.fit(x_train - mu_train).covariance_).float()

    p_test = lambda x: p_gauss(x, mu_test, S2_test)
    p_train = lambda x: p_gauss(x, mu_train, S2_train)
    w_hat = torch.vmap(p_test)(x_train) / torch.vmap(p_train)(x_train)
    return w_hat[:, 0]


# Kernel mean matching to estimate the ratio of p_test / p_train
# Lambda corresponds to kernel inverse regularizer
def kernel_mean_matching(x_test, x_train, normalize='False', bound=10.0, eps_scale=0.01, lamb=1e-3):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    # Set sigma to the median distance between all samples
    dist_x = torch.cdist(x_train, x_train)
    sigma = dist_x.median()
    # Construct Kernel matrices between train and test
    ker_mat = torch.exp(-(dist_x**2) / (2*(sigma**2)))
    ker_vec = torch.sum(torch.exp(-(torch.cdist(x_train, x_test)**2) / (2*(sigma**2))), dim=1) * (num_train / num_test)

    eps = eps_scale * bound / np.sqrt(num_train)
    G = ker_mat.double().numpy()
    G += lamb * np.eye(num_train)
    a = ker_vec.double().numpy()
    # Normalization seems to make test error performance worse
    if normalize is True:
        C = torch.cat((torch.ones(1, num_train), -torch.ones(1, num_train), torch.eye(num_train), -torch.eye(num_train))).double().numpy().T
        b = torch.cat((torch.tensor([(1 - eps) * num_train, -(1 + eps) * num_train]), torch.zeros(num_train), -bound * torch.ones(num_train))).double().numpy()
    else:
        C = torch.cat((torch.eye(num_train), -torch.eye(num_train))).double().numpy().T
        b = torch.cat((torch.zeros(num_train), -bound * torch.ones(num_train))).double().numpy()
    w_hat = quadprog.solve_qp(G, a, C, b)[0]
    w_hat = np.maximum(np.zeros(num_train), w_hat)
    return torch.tensor(w_hat)
