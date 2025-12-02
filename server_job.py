from mc_computations import *
from example_with_maximum import exported_parameters
import os
import matplotlib.pyplot as plt

exported_parameters["resistance"] = np.array([[140/189, 1],[1, 39/57]])
exported_parameters["cost_test"] = 8

delta = 0.06
starting_val = 0.9
initial_direction = -1  # -1 means start by decreasing
prev_seq = [starting_val+initial_direction*i*delta for i in range(0, 6)]
extra_name = f"_cost_{exported_parameters["cost_test"]}"
backward = [starting_val-initial_direction*(i-6)*delta for i in range(0, 7)]
prev_seq.extend(backward)

if initial_direction == -1:
    experiment_name = f"old_resistance_temporal_starting_val_{starting_val}_delta_{delta}_decreasing"
else:
    experiment_name = f"old_resistance_temporal_starting_val_{starting_val}_delta_{delta}_increasing"

if extra_name is not None:
    experiment_name = experiment_name + extra_name

# this is only needed to save the results of the priors
save_to = os.path.join(os.getcwd(),"diagnostic_value","longer_runs")

# the above is automatically included in the folder name for the experiment, so only this
# needs to be added
subpath = os.path.join("average_optimum_runs",experiment_name,f"decay_rate_{5/1000:.4f}")

run_optimal_average_experiment(experiment_name, prev_seq, local_params=exported_parameters, prior_decay_rate=5/1000)
# obj_vals = np.load(os.path.join(save_to, subpath, "avg_objective_values.npy"))
# p_vals = np.linspace(0,1,1001)
# plt.plot(p_vals, obj_vals)
# plt.show()
