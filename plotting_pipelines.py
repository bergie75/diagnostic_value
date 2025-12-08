from mc_computations import *
from example_with_maximum import exported_parameters
import os
import matplotlib.pyplot as plt

def one_frequency_temporal_plots(delta_list, local_parameters=exported_parameters, plot_labels=None):
    local_parameters["resistance"] = np.array([[140/189, 1],[1, 39/57]])
    local_parameters["cost_test"] = 6

    if plot_labels is None:
        plot_labels = [f"{x}" for x in delta_list]

    # information to find related experiments
    starting_val = 0.9
    initial_direction = -1  # -1 means start by decreasing
    extra_name = f"_cost_{local_parameters["cost_test"]}"

    p_vals = np.linspace(0,1,local_parameters["num_patients"]+1)
    optimal_testing_rates = []

    for delta in delta_list:
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

        avg_obj_vals = np.load(os.path.join(save_to, subpath, "avg_objective_values.npy"))
        max_index = np.argmax(avg_obj_vals)
        optimal_testing_rates.append(p_vals[max_index]*100)
    
    plt.bar(plot_labels, optimal_testing_rates)
    plt.xlabel("rate of change of disease prevalence")
    plt.ylabel("optimal fraction of patients to test (%)")
    plt.show()
    
if __name__ == "__main__":
    one_frequency_temporal_plots([0.015, 0.03, 0.06], plot_labels=["0.5x baseline", "baseline", "2x baseline"])