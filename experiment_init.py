import numpy as np
from scipy.linalg import expm
from utils import normalize
import logging
logger = logging.getLogger(__name__)

class HarmonicOscillator:
    def __init__(self, L, m, omega, nbins, total_time, taylor_order, wavefuntion, initial_momentum = 0, mu=0, sigma=1):
        self.L = L
        self.m = m
        self.omega = omega
        self.nbins = nbins ### want some auto way to calculate this 
        self.delta_t = 0.01 #again, would like some automatic way to calculate this
        self.taylor_order = taylor_order
        self.delta_x = L/nbins
        self.x0 = np.linspace(-L/2, L/2, nbins)
        self.wavefunction = wavefuntion
        self.p0 = initial_momentum
        self.mu = mu
        self.sigma = sigma
        assert self.wavefunction in ["gaussian", "eigenfunc"], "invalid wavefunction selection"
    
    def configure_initial_state(self):
        psi = np.zeros_like(self.x0)
        if self.wavefunction == "energy_eigenfunc":
            psi = np.array((self.m*self.omega/np.pi)**0.25 * np.exp(-0.5*self.m*self.omega*self.x0**2), dtype=complex)
        elif self.wavefunction == "gaussian":
            psi = np.array(np.exp(- (self.x0 - self.mu)**2 / (2 * self.sigma**2)) * np.exp(1j * self.p0 * self.x0), dtype=complex)
        logger.info("Initial wavefunction set")
        psi_normed = normalize(psi, self.delta_x)
        return psi_normed
    
    def get_time_evolution_operator(self):
        Derivative_squared = np.diag([-2]*self.nbins)
        Derivative_squared += np.diag([1]*(self.nbins-1), 1) 
        Derivative_squared += np.diag([1]*(self.nbins-1), -1)
        Derivative_squared[0, -1] = 1
        Derivative_squared[-1, 0] = 1
        
        X = np.diag(self.x0)

        H = ((-1/(self.delta_x**2 * 2 * self.m)) * Derivative_squared) + ((1/2) * (self.m) * (self.omega**2) * (X)**2)
        
        time_evolution_operator = expm(-1j * H * self.delta_t)

        logger.info("Computed time evolution operator")
        return time_evolution_operator
    
    def get_setup(self):
        """A method that returns a string describing the experimental setup of the QHO"""
        
        return f"""QHO with L = {self.L}, mass = {self.m}, omega = {self.omega}, expanding the hamiltonian to taylor order 
    {self.taylor_order}. In starting state: {self.wavefunction}."""
        
    
        
    
        