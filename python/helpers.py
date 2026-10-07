import numpy as np

def rmse(error):
    """
    Compute the root mean square error (RMSE) of a given error array.
    """
    return np.sqrt(np.nanmean(error ** 2))