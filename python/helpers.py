import numpy as np
from scipy.stats import chi2

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

def anees_bounds(n_runs: int, n_states: int, confidence: float):
    """
    Confidence interval for ANEES. N * ANEES is chi-squared with N * n degrees of freedom.
    [1, Eq. 9.13]
    """
    alpha = 1 - confidence
    r1 = chi2.ppf(alpha / 2, n_runs * n_states) / n_runs
    r2 = chi2.ppf(1 - alpha / 2, n_runs * n_states) / n_runs
    return r1, r2
