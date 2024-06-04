import numpy as np


def gaussian_entropy(std: np.ndarray) -> np.ndarray:
    """Compute the entropy of a Gaussian distribution given standard deviation.

    Parameters
    ----------
    std : np.ndarray
        Standard deviation array.

    Returns
    -------
    entropy: np.ndarray
        Entropy array.

    """
    entropy = 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5
    return entropy

def gaussian_entropy_multivariate(K: np.ndarray):
    """Compute the entropy of a Gaussian distribution given standard deviation.

    Parameters
    ----------
    K : np.ndarray
        covariance matrix array.

    Returns
    -------
    entropy: float.

    """
    K_det = np.linalg.det(K)
    if K_det <= 0:
        K_det = 0.00000001
    entropy = 0.5 * np.log((2 * np.pi)**K.shape[0]*K_det)
    return entropy
