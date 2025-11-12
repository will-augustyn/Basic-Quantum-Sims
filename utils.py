import numpy as np

def normalize(psi:np.array, dx:float):
    """return a normalized wavefunction

    Args:
        psi (np.array): numpy array (comlex and/or real values)
        dx (float): spacing between x values
    """
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm