import numpy as np
from scipy.linalg import expm
import utils
import logging
logger = logging.getLogger(__name__)

class HarmonicOscillator:
    def __init__(self, L, m, omega, taylor_order, wavefuntion, initial_momentum = 0, mu=0, sigma=1):
        self.L = L
        self.m = m
        self.omega = omega
        self.delta_t = 0.01 #again, would like some automatic way to calculate this
        self.taylor_order = taylor_order
        self.wavefunction = wavefuntion
        self.p0 = initial_momentum
        self.mu = mu
        self.sigma = sigma
        self.nbins, self.delta_x = utils.set_nbins(self.p0, self.m, self.mu**2, self.L)
        self.x0 = np.linspace(-self.L/2, self.L/2, self.nbins)
    
    def configure_initial_state(self):
        psi = np.zeros_like(self.x0)
        if self.wavefunction == "gaussian":
            psi = np.array(np.exp(- (self.x0 - self.mu)**2 / (2 * self.sigma**2)) * np.exp(1j * self.p0 * self.x0), dtype=complex)
        logger.info("Initial wavefunction set")
        psi_normed = utils.normalize(psi, self.delta_x)
        return psi_normed
    
    def get_time_evolution_operator(self):
        Derivative_squared = utils.get_derivative_squared(self.nbins, periodic=True)
        
        X = np.diag(self.x0)

        H = ((-1/(self.delta_x**2 * 2 * self.m)) * Derivative_squared) + ((1/2) * (self.m) * (self.omega**2) * (X)**2)
        
        time_evolution_operator = expm(-1j * H * self.delta_t)

        logger.info("Computed time evolution operator")
        return time_evolution_operator
    
    def get_setup(self):
        """A method that returns a string describing the experimental setup of the QHO"""
        
        return f"""QHO with L = {self.L}, mass = {self.m}, omega = {self.omega}, expanding the hamiltonian to taylor order 
    {self.taylor_order}. In starting state: {self.wavefunction}."""
    
    def run_checks(self):
        assert self.wavefunction in ["gaussian"], "invalid wavefunction selection"
        assert utils.estimate_intial_energy(p0=self.p0, m=self.m, V_x0=self.mu**2) < np.max(self.x0)**2, "initial energy greater than max potential"
        if self.wavefunction == "gaussian":
            assert self.delta_x < (self.sigma / 20), "delta x is greater than length scale of initial wavefunction"
    
class ParticleInABox:
    def __init__(self, L, m, taylor_order, wavefuntion, initial_momentum = 0, mu=0, sigma=1, n=1):
        self.L = L
        self.m = m
        self.delta_t = 0.01 #again, would like some automatic way to calculate this
        self.taylor_order = taylor_order
        self.wavefunction = wavefuntion
        self.n = n
        self.p0 = initial_momentum
        self.mu = mu
        self.sigma = sigma
        self.nbins, self.delta_x = utils.set_nbins(self.p0, self.m, 0, self.L)
        print(self.nbins)
        self.x0 = np.linspace(0, self.L, self.nbins)

    def configure_initial_state(self):
        psi = np.zeros_like(self.x0)
        if self.wavefunction == "eigenfunc":
            psi = np.array(np.sqrt(2/self.L)*np.sin((self.n*np.pi*self.x0)/self.L), dtype=complex)
        elif self.wavefunction == "gaussian":
            psi = np.array(np.exp(- (self.x0 - self.mu)**2 / (2 * self.sigma**2)) * np.exp(1j * self.p0 * self.x0), dtype=complex)
        logger.info("Initial wavefunction set")
        psi_normed = utils.normalize(psi, self.delta_x)
        return psi_normed

    def get_time_evolution_operator(self):
        Derivative_squared = utils.get_derivative_squared(self.nbins, False)

        H = ((-1/(self.delta_x**2 * 2 * self.m)) * Derivative_squared)
        
        time_evolution_operator = expm(-1j * H * self.delta_t)

        logger.info("Computed time evolution operator")
        return time_evolution_operator

    def get_setup(self):
        """A method that returns a string describing the experimental setup"""
        
        state =  f"""Particle in a box with L = {self.L}, mass = {self.m}, expanding the hamiltonian to taylor order 
    {self.taylor_order}. In starting state: {self.wavefunction}"""
        if self.wavefunction == "eigenfunc":
            state += f"with n = {self.n}"
        return state
    
    def run_checks(self):
        assert self.wavefunction in ["gaussian", "eigenfunc"], "invalid wavefunction selection"
        if self.wavefunction == "gaussian":
            assert self.delta_x < (self.sigma / 20), "delta x is greater than length scale of initial wavefunction"
    
class Tunneling:
    def __init__(self, L, m, taylor_order, wavefunction, potential_width, potential_height, potential_offset, initial_momentum=0, mu=0, sigma=1):
        self.L = L 
        self.m = m
        self.delta_t = 0.01 #again, would like some automatic way to calculate this
        self.taylor_order = taylor_order
        self.wavefunction = wavefunction
        self.p0 = initial_momentum
        self.mu = mu
        self.sigma = sigma
        self.nbins, self.delta_x = utils.set_nbins(self.p0, self.m, 0, self.L)
        self.x0 = np.linspace(0, L, self.nbins)
        self.psi = np.zeros(self.nbins)
        
        self.potential_height = potential_height
        self.potential_offset = potential_offset #distance from left (x=0) potential bump should begin at
        self.potential_width = potential_width
        self.potential_array = np.zeros(self.nbins)
        
    def configure_initial_state(self):
        allowed_initial_state_x0 = np.linspace(0, self.potential_offset, int((self.nbins*self.potential_offset)/self.L)) 
        #need to ensure starting wavefunction is purely in 'allowed' region 
        psi = np.zeros_like(allowed_initial_state_x0)
        if self.wavefunction == "gaussian":
            psi = np.array(np.exp(- (allowed_initial_state_x0 - self.mu)**2 / (2 * self.sigma**2)) * np.exp(1j * self.p0 * allowed_initial_state_x0), dtype=complex)
        psi = np.append(psi, np.zeros(self.nbins-len(psi)))
        #pad in 0s to get back to correct length 
        logger.info("Initial wavefunction set")
        psi_normed = utils.normalize(psi, self.delta_x)
        self.psi = psi_normed
        return psi_normed
    
    def get_time_evolution_operator(self):
        Derivative_squared = utils.get_derivative_squared(self.nbins, periodic=False)
        
        potential_array = [0] * int(self.nbins * self.potential_offset/self.L)
        potential_array += ([self.potential_height] * int(self.nbins * self.potential_width/self.L))
        pad_length = self.nbins - len(potential_array)
        potential_array += [0] * pad_length
        self.potential_array = potential_array
        
        V = np.diag(potential_array)
        
        H = ((-1/(self.delta_x**2 * 2 * self.m)) * Derivative_squared) + V
        
        energy = utils.estimate_intial_energy(self.p0, self.m, 0)
        
        if energy >= self.potential_height:
            logger.warning("intial wavefunction has energy higher than potential bump")
        
        time_evolution_operator = expm(-1j * H * self.delta_t)

        logger.info("Computed time evolution operator")
        return time_evolution_operator
    
    def get_setup(self):
        """A method that returns a string describing the experimental setup"""
        state =  f"""Tunneling experiment with L = {self.L}, mass = {self.m}, expanding the hamiltonian to taylor order 
    {self.taylor_order}. In starting state: {self.wavefunction}."""
        return state
    
    def run_checks(self):
        assert self.wavefunction in ["gaussian"], "invalid wavefunction selection"
        assert self.delta_x < (self.sigma / 20), "delta x is greater than length scale of initial wavefunction"
        assert self.potential_width < self.L, "potential width can't be greater than L"
        assert utils.estimate_intial_energy(p0=self.p0, m=self.m, V_x0=0) < self.potential_height, "initial energy greater than max potential"