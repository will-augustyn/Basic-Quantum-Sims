import numpy as np
import matplotlib.pyplot as plt 
import pickle
from matplotlib.animation import FuncAnimation
import logging
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, x0, save_path, fps=30, data_path=None, data=None, playback_speed = 1):
        """intitliazer the visualizer. Note: must pass in either a data_path or data. Can't leave both as None.

        Args:
            x0 (np.array): numpy array containing the x bins
            save_path (str): where to save video animations to 
            fps (int, optional): the frames per second for the animation. Defaults to 30.
            data_path (str, optional): where the data to visulaize is stored (must be a pickle file). Defaults to None.
            data (np.array, optional): where to pass in data if not saving on disk. Defaults to None.
        """
        self.data = data 
        self.x0 = x0
        self.save_path = save_path
        self.fps = fps
        self.playback_speed = playback_speed
        if data_path:
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = data
        
    def create_basic_animation_psi_x_real(self, include_potential=False, potential_array=None):
        fig, ax = plt.subplots()
        max_val = np.max(np.abs(np.real(self.data)))
        if include_potential:
            ax.set_xlim(np.min(self.x0), np.max(self.x0))
            ax.set_ylim(0, 1.5*np.max(potential_array))
        else:
            ax.set_xlim(np.min(self.x0), np.max(self.x0))
            ax.set_ylim(0, 1.1 * max_val)
        ax.set_xlabel("x")
        ax.set_ylabel("Re(ψ)")
        
        if include_potential:
            ax.plot(self.x0, potential_array, color = "orange", label="V(x)")
        
        line, = ax.plot([], [], lw=2)
        total_time_steps = len(self.data)  
        
        def init():
            line.set_data([], [])
            return (line,)
        
        def update(i):
            y = np.real(self.data[i])
            line.set_data(self.x0, y)
            return (line,)
        
        anim = FuncAnimation(fig, update, frames=range(0, total_time_steps, self.playback_speed), init_func=init)
        anim.save(self.save_path, writer="ffmpeg", fps=self.fps)
        plt.close(fig)
        
        logger.info(f"Created basic simulation of the real part of psi at {self.fps} fps")
        return
    
    def create_basic_animation_probability_density(self, include_potential=False, potential_array=None):
        fig, ax = plt.subplots()
        max_val = np.max(np.abs(np.real(self.data)))
        if include_potential:
            ax.set_xlim(np.min(self.x0), np.max(self.x0))
            ax.set_ylim(-1.1 * max_val, 1.5*np.max(potential_array))
        else:
            ax.set_xlim(np.min(self.x0), np.max(self.x0))
            ax.set_ylim(-1.1 * max_val, 1.1 * max_val)
        ax.set_xlabel("x")
        ax.set_ylabel("$|\psi(x)|^2$")
        
        if include_potential:
            ax.plot(self.x0, potential_array, color = "orange", label="V(x)")
        
        line, = ax.plot([], [], lw=2)
        total_time_steps = len(self.data)  
        
        def init():
            line.set_data([], [])
            return (line,)
        
        def update(i):
            y = np.real(np.abs(self.data[i])**2)
            line.set_data(self.x0, y)
            return (line,)
        
        anim = FuncAnimation(fig, update, frames=range(0, total_time_steps, self.playback_speed), init_func=init)
        anim.save(self.save_path, writer="ffmpeg", fps=self.fps)
        plt.close(fig)
        
        logger.info(f"Created basic simulation of psi sqaured at {self.fps} fps")
        return