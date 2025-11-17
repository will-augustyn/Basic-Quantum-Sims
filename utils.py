import numpy as np

def normalize(psi:np.array, dx:float):
    """return a normalized wavefunction

    Args:
        psi (np.array): numpy array (comlex and/or real values)
        dx (float): spacing between x values
    """
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm

def get_derivative_squared(nbins, periodic):
    """returns the derivative operator (periodic or fixed)

    Args:
        nbins (int): the number of x bins in the experiment
        periodic (bool): True for periodic boundary conditions, False for fixed

    Returns:
        np.array: matrix representation of the derivative operator
    """
    Derivative_squared = np.diag([-2]*nbins)
    Derivative_squared += np.diag([1]*(nbins-1), 1) 
    Derivative_squared += np.diag([1]*(nbins-1), -1)
    if periodic:
        Derivative_squared[0, -1] = 1
        Derivative_squared[-1, 0] = 1
    return Derivative_squared

def compute_expected_value(operator, state):
    """compute the expecation value of an operator in a state

    Args:
        operator (np.array): an operator in matrix form
        state (np.array): vector the expectation shall be computed under

    Returns:
        float: expected value of operator in state 
    """
    return state @ (operator @ state)

def estimate_intial_energy(p0, m, V_x0):
    """estimates the initial energy of a state

    Args:
        p0 (float): initial momentum
        m (float): particle mass
        V_x0 (float): estimate of initial potential energy

    Returns:
        float: estimate of initial energy
    """
    return ((p0**2)/2*m) + V_x0

def set_nbins(p0, m, V_x0, L):
    """sets the number of x bins that should be used for an experiment

    Args:
        p0 (float): initial momentum
        m (float): mass
        V_x0 (float): estimate of initial potential energy
        L (float): experiment length 

    Returns:
        float: nbins is total number of bins, dx is spacing between them 
    """
    if p0 == 0:
        nbins = int(L/0.01)
        dx = L/nbins
        return nbins, dx
    E0 = estimate_intial_energy(p0, m, V_x0)
    p_max = np.sqrt(2*m*E0)
    min_wavelength = 1/p_max
    dx = 0.01 * min_wavelength
    nbins = int(L / dx)
    return nbins, dx