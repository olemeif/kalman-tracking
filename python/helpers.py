import numpy as np

def rmse(error):
    """
    Compute the root mean square error (RMSE) of a given error array.
    """
    return np.sqrt(np.nanmean(error ** 2))

def nees(error, P):
    """
    Normalized estimation error squared, [1] Eq. 9.7: e^T P^-1 e.
    Works on any leading shape, e.g. error (n_runs, n_steps, n) and P (n_runs, n_steps, n, n).
    """
    return np.einsum("...i,...i->...", error, np.linalg.solve(P, error[..., None])[..., 0])
