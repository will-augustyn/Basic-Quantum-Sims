import logging, sys

from experiment_init import HarmonicOscillator, ParticleInABox, Tunneling
from conduct_experiment import RunExperiment
from visualize_results import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger(__name__)

experiment_list = ["QHO", "ParticleInBox", "Tunneling"]

def main(experiment):
    logger.info("Starting simulation")
    assert experiment in experiment_list, f"invalid experiment, please choose from: {experiment_list}"
    if experiment == "QHO":
        my_experiment_setup = HarmonicOscillator(10, 1, 1, 20, "gaussian", mu=1)
        logger.info(my_experiment_setup.get_setup())
        my_experiment_setup.run_checks()
        psi0 = my_experiment_setup.configure_initial_state()
        time_ev_op = my_experiment_setup.get_time_evolution_operator()
        
        run_experiment = RunExperiment(psi0, time_ev_op, 10, my_experiment_setup.delta_t, my_experiment_setup.delta_x)
        data = run_experiment.time_evolve()
        vis = Visualizer(my_experiment_setup.x0, "/Users/willaugustyn/QuantumSims/animations/QHO/test.mp4", data=data)
        vis.create_basic_animation_psi_x_real(include_potential=True, potential_array=my_experiment_setup.x0**2)
    elif experiment == "ParticleInBox":
        my_experiment_setup = ParticleInABox(L=10, m=1, taylor_order=20, wavefuntion="eigenfunc", n=3)
        logger.info(my_experiment_setup.get_setup())
        my_experiment_setup.run_checks()
        psi0 = my_experiment_setup.configure_initial_state()
        time_ev_op = my_experiment_setup.get_time_evolution_operator()
        
        run_experiment2 = RunExperiment(psi0, time_ev_op, 10, my_experiment_setup.delta_t, my_experiment_setup.delta_x)
        data = run_experiment2.time_evolve()
        vis = Visualizer(my_experiment_setup.x0, "/Users/willaugustyn/QuantumSims/animations/particle_in_box/test_PIB_1x.mp4", data=data)
        vis.create_basic_animation_psi_x_real()
    elif experiment == "Tunneling":
        my_experiment_setup = Tunneling(15, 1, 20, "gaussian", potential_width=2, potential_height=5, potential_offset=3, 
                                        initial_momentum=2, mu=1, sigma=0.3)
        logger.info(my_experiment_setup.get_setup())
        my_experiment_setup.run_checks()
        psi0 = my_experiment_setup.configure_initial_state()
        time_ev_op = my_experiment_setup.get_time_evolution_operator()
        
        run_experiment3 = RunExperiment(psi0, time_ev_op, 10, my_experiment_setup.delta_t, my_experiment_setup.delta_x)
        data = run_experiment3.time_evolve()
        vis = Visualizer(my_experiment_setup.x0, "/Users/willaugustyn/QuantumSims/animations/tunneling/test_tunneling_1x.mp4", data=data)
        vis.create_basic_animation_probability_density(include_potential=True, potential_array=my_experiment_setup.potential_array)
        
if __name__ == '__main__':
    main("Tunneling")