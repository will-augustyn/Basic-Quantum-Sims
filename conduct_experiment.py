import numpy as np 
import pickle
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

from utils import normalize

class HarmonicOscillatorRunExperiment:
    def __init__(self, psi0, time_evolution_operator, total_runtime, delta_t, delta_x, save_locally = False):
        self.psi0 = psi0
        self.time_ev_op = time_evolution_operator
        self.total_runtime = total_runtime
        self.delta_t = delta_t
        self.save_locally = save_locally
        self.delta_x = delta_x
        
        self.nsteps = int(self.total_runtime / delta_t)
        
    def time_evolve(self):
        """Create a list that stores the time evolved states"""
        list_of_time_evolved_states = [self.psi0]
        t = 0
        psi = self.psi0
        for t_step in tqdm(range(self.nsteps), desc="Evolving wavefunction", ncols=80):
            psi = self.time_ev_op @ psi
            psi = normalize(psi, self.delta_x)
            list_of_time_evolved_states.append(psi)
            t += self.delta_t    
        logger.info("Finished time evolving")
        if self.save_locally == True:
            with open('/Users/willaugustyn/QuantumSims/data/time_ev_data.pkl', 'wb') as f:
                pickle.dump(list_of_time_evolved_states, f)
            return
        return list_of_time_evolved_states