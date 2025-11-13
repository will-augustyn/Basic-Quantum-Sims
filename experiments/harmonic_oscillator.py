import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

#parameters
L = 10
nbins = 1000
delta_x = L/nbins
n=1 
delta_t = 0.01
m=1
taylor_order = 20
total_time = 1000
omega = 1
x0 = np.linspace(-L/2, L/2, nbins) #initial x 

def main(x0_wavefunction, visualize, save=True, save_name = ''):
    print(f"running QHO with L = {L}, m = {m}, with starting wavefunction: {x0_wavefunction}")
    psi_x = set_wavefunction(x0_wavefunction)
    if visualize == "psi_squared":
        time_evolve_psi_squared(psi_x, save, save_name)
    elif visualize == "psi_x":
        time_evolve_psi_x(psi_x, save, save_name)

def set_wavefunction(type):
    psi = np.zeros_like(x0)
    if type == "energy_eigenfunc":
        psi = np.array([(m*omega/np.pi)**0.25 * np.exp(-0.5*m*omega*x**2) for x in x0], dtype=complex)
    elif type == "gaussian":
        mu = 1
        sigma = 1
        p0 = 0
        psi = np.array(np.exp(- (x0 - mu)**2 / (2 * sigma**2)) * np.exp(1j * p0 * x0), dtype=complex)
    print("wavefunction set")
    psi_normed = normalize(psi)
    return psi_normed

def normalize(psi):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * delta_x)
    return psi / norm

def time_evolve_psi_squared(psi, save, save_name):
    D2 = np.diag([-2]*nbins)
    D2 += np.diag([1]*(nbins-1), 1) 
    D2 += np.diag([1]*(nbins-1), -1)
    D2[0, -1] = 1
    D2[-1, 0] = 1
    
    X = np.diag([x**2 for x in x0])

    H = ((-1/(delta_x**2 * 2*m)) * D2) + (1/2) * (m) * (omega**2) * (X)

    time_ev_op = expm(-1j * H * delta_t)

    ##Update part
    fig, ax = plt.subplots()
    line, = ax.plot(x0, np.zeros_like(x0), lw=2, label=r'$|\psi(x,t)|^2$')
    ax.set(ylim=(-2, 2),
        xlabel='x', ylabel=r'$|\psi(x,t)|^2$',
        title='QHO: Probability Density')
    ax.legend(loc='upper right')

    steps = int(np.ceil(total_time / delta_t))
    frame_skip = 50  # increase to speed up animation by skipping frames

    def init():
        y = np.abs((psi))**2
        line.set_ydata(y)
        return (line,)

    def update(frame):
        # time step: interior only
        nonlocal psi
        #print(psi)
        psi = time_ev_op @ psi
        # renormalize occasionally to tame Taylor truncation drift
        if (frame + 1) % 1 == 0:
            psi = normalize(psi)
        y = np.abs((psi))**2
        line.set_ydata(y)
        return (line,)

    ani = FuncAnimation(
        fig, update, init_func=init,
        frames=range(0, steps, frame_skip),
        interval=30, blit=True
    )
    if save:
        ani.save(f"{save_name}", writer="ffmpeg", fps=int(1/delta_t/5))
        print("Animation saved")
    
def time_evolve_psi_x(psi, save, save_name):
    D2 = np.diag([-2]*nbins)
    D2 += np.diag([1]*(nbins-1), 1) 
    D2 += np.diag([1]*(nbins-1), -1)
    D2[0, -1] = 1
    D2[-1, 0] = 1
    
    X = np.diag(x0)

    H = ((-1/(delta_x**2 * 2*m)) * D2) + ((1/2) * (m) * (omega**2) * (X)**2)
    
    print(((1/2) * (m) * (omega**2) * (X)**2))

    time_ev_op = expm(-1j * H * delta_t)

    ##Update part
    fig, ax = plt.subplots()
    line, = ax.plot(x0, np.zeros_like(x0), lw=2, label=r'$\psi(x,t)$')
    ax.set(ylim = (-2, 2),
        xlabel='x', ylabel=r'$\psi(x,t)$',
        title='QHO: Wavefunction')
    ax.legend(loc='upper right')
    
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = plt.get_cmap('hsv')

    # Build initial colored fill under the curve using a PolyCollection
    y0 = psi
    phase0 = np.angle(psi)
    verts = [ [(x0[i], 0.0),
               (x0[i+1], 0.0),
               (x0[i+1], y0[i+1]),
               (x0[i], y0[i])] for i in range(nbins-1) ]
    # color each vertical strip by the phase at the left edge (could also use midpoint)
    colors = cmap(norm(phase0[:-1]))
    fill = PolyCollection(verts, facecolors=colors, edgecolor='none', zorder=1)
    ax.add_collection(fill)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("phase arg(ψ) [rad]")

    steps = int(np.ceil(total_time / delta_t))
    frame_skip = 50  # increase to speed up animation by skipping frames

    def init():
        y = np.real(psi)
        line.set_ydata(y)
        return (line,)

    def update(frame):
        # time step: interior only
        nonlocal psi
        #print(psi)
        psi = time_ev_op @ psi
        # renormalize occasionally to tame Taylor truncation drift
        if (frame + 1) % 1 == 0:
            psi = normalize(psi)
        y = np.real(psi)
        line.set_ydata(y)
        
        ax.relim()
        ax.autoscale_view()
        
        phase = np.angle(psi)
        new_verts = [ [(x0[i], 0.0),
                       (x0[i+1], 0.0),
                       (x0[i+1], y[i+1]),
                       (x0[i], y[i])] for i in range(nbins-1) ]
        fill.set_verts(new_verts)
        fill.set_facecolors(cmap(norm(phase[:-1])))
        
        return (line, fill)

    ani = FuncAnimation(
        fig, update, init_func=init,
        frames=range(0, steps, frame_skip),
        interval=30, blit=True
    )
    if save:
        ani.save(f"{save_name}", writer="ffmpeg", fps=int(1/delta_t))
        print("Animation saved")
        
        

