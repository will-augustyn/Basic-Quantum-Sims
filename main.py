import logging, sys

from experiment_init import HarmonicOscillator
from conduct_experiment import HarmonicOscillatorRunExperiment
from visualize_results import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger(__name__)

experiment_list = ["QHO"]

def main(experiment):
    logger.info("Starting simulation")
    assert experiment in experiment_list, f"invalid experiment, please choose from: {experiment_list}"
    if experiment == "QHO":
        my_experiment_setup = HarmonicOscillator(10, 1, 1, 1000, 100, 20, "gaussian", mu=1)
        logger.info(my_experiment_setup.get_setup())
        psi0 = my_experiment_setup.configure_initial_state()
        time_ev_op = my_experiment_setup.get_time_evolution_operator()
        run_experiment = HarmonicOscillatorRunExperiment(psi0, time_ev_op, 100, my_experiment_setup.delta_t, my_experiment_setup.delta_x)
        data = run_experiment.time_evolve()
        vis = Visualizer(data, my_experiment_setup.x0, "/Users/willaugustyn/QuantumSims/data/test.mp4")
        vis.create_basic_animation_psi_x_real()
        
if __name__ == '__main__':
    main("QHO")