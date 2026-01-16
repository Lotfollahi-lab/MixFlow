

"""
Calculates the MMD metric as done by Klein et al: https://github.com/theislab/CellFlow
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from collections.abc import Sequence
import time

from scipy import sparse
from scipy.sparse import coo_array

import torch


@torch.no_grad()
def rbf_kernel_torch(x, y, gamma: float):
    assert isinstance(x, np.ndarray)
    assert isinstance(x, np.ndarray)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ten_x = torch.tensor(x + 0.0).float().to(device)  # [N x D]
    ten_y = torch.tensor(y + 0.0).float().to(device)  # [M x D]

    xx = (ten_x * ten_x).sum(1).unsqueeze(1)  # [N x 1]
    yy = (ten_y * ten_y).sum(1).unsqueeze(0)  # [1 x M]
    xy = torch.matmul(ten_x, ten_y.T)  # [N x M]
    sq_distances = xx + yy - 2 * xy

    assert gamma > 0

    return torch.exp(-gamma * sq_distances).detach().cpu().numpy()  # [N x M]


def maximum_mean_discrepancy(x, y, gamma: float, mode_comp_kernel: str) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two distributions x and y.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        gamma
            Parameter for the rbf kernel.
        mode_comp_kernel
            A string in ['scipy', 'jax', 'torch'].

    Returns
    -------
        A scalar denoting the squared maximum mean discrepancy loss.
    """
    # kernel = rbf_kernel if exact else rbf_kernel_fast
    kernel = {
        'scipy':rbf_kernel,
        'jax':None,
        'torch':rbf_kernel_torch
    }[mode_comp_kernel]

    # print("Computing MMD in mode '{}' ".format(mode_comp_kernel))

    xx = kernel(x, x, gamma)
    xy = kernel(x, y, gamma)
    yy = kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()



def compute_scalar_mmd(x, y, mode_comp_kernel, gammas: Sequence[float] | None = None) -> float:
    """Compute the Mean Maximum Discrepancy (MMD) across different length scales

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        gammas
            A sequence of values for the paramater gamma of the rbf kernel.

    Returns
    -------
        A scalar denoting the average MMD over all gammas.
    """

    assert gammas is None  # as often used in cellflow_reproducbility

    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    mmds = [maximum_mean_discrepancy(x, y, gamma=gamma, mode_comp_kernel=mode_comp_kernel) for gamma in gammas]  # type: ignore[union-attr]
    return np.nanmean(np.array(mmds))


def iface_compute_MMD(x:torch.Tensor, y:torch.Tensor):
    """
    Compatible the with the call made by `./metrics.py`
    """
    return compute_scalar_mmd(
        x.detach().cpu().numpy()+0.0,
        y.detach().cpu().numpy()+0.0,
        mode_comp_kernel='torch',
        gammas=None
    )



    

