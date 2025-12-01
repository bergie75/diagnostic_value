from mc_computations import *
from example_with_maximum import exported_parameters
import os
import matplotlib.pyplot as plt

delta = 0.015
starting_val = 0.9
initial_direction = -1  # -1 means start by decreasing
prev_seq = [starting_val+initial_direction*i*delta for i in range(0, 6)]
extra_name = None
backward = [starting_val-initial_direction*(i-6)*delta for i in range(0, 7)]
prev_seq.extend(backward)

if initial_direction == -1:
    experiment_name = f"old_resistance_temporal_starting_val_{starting_val}_delta_{delta}_decreasing"
else:
    experiment_name = f"old_resistance_temporal_starting_val_{starting_val}_delta_{delta}_increasing"

if extra_name is not None:
    experiment_name = experiment_name + extra_name

exported_parameters["resistance"] = np.array([[140/189, 1],[1, 39/57]])
#run_temporal_experiment(experiment_name, prev_seq, local_params=exported_parameters)

run_optimal_average_experiment(experiment_name, prev_seq, local_params=exported_parameters, prior_decay_rate=5/1000)
