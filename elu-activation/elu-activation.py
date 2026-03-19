import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    x = np.array(x)
    result = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return result.tolist()