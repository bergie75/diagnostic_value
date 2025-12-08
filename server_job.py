from mc_computations import *
from example_with_maximum import exported_parameters
import os
import matplotlib.pyplot as plt

exported_parameters["resistance"] = np.array([[140/189, 1],[1, 39/57]])
exported_parameters["cost_test"] = 6

# where does the disease prevalence start and end
starting_val = 0.9
ending_val = 0.54

num_time_periods = [7,13,25]
for num_months in num_time_periods:
    # we reach the same start and end prevalences, just so only difference is the rates
    prev_seq = np.linspace(starting_val, ending_val, num_months)
    extra_name = f"_cost_{exported_parameters["cost_test"]}"

    experiment_name = f"starting_val_{starting_val}_ending_val_{ending_val}"

    if extra_name is not None:
        experiment_name = experiment_name + extra_name

    run_optimal_average_experiment(experiment_name, prev_seq, local_params=exported_parameters, prior_decay_rate=5/1000)

# obj_vals = np.load(os.path.join(save_to, subpath, "avg_objective_values.npy"))
# p_vals = np.linspace(0,1,1001)
# plt.plot(p_vals, obj_vals)
# plt.show()
