from mc_computations import *
from philipson_setup import *
from example_with_maximum import exported_parameters
import os
import matplotlib.pyplot as plt
from copy import deepcopy

def one_frequency_temporal_plots(delta_list, local_parameters=deepcopy(exported_parameters), plot_labels=None):
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
    
def philipson_plots(xmin=0, xmax=20, price=13, dist_params = [[15,1.7], [10,3]]):
    gaussian = lambda x,mu,sigma: 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
    full_domain = np.linspace(xmin, xmax, 1000)
    right_of_price = np.linspace(price, xmax, 1000)

    funcs = [(lambda x, par=par: gaussian(x, par[0], par[1])) for par in dist_params]
    top_heights = [f(price) for f in funcs]
    
    plt.vlines(price, 0, max(top_heights), linestyles='dashed', colors='black', label="diagnostic price")
    for f in funcs:
        plt.plot(full_domain, f(full_domain))
        plt.fill_between(right_of_price, f(right_of_price), alpha=0.3)
    plt.ylabel("density of patients")
    plt.xlabel("cost of empirical treatment")
    plt.legend()
    plt.show()

def comparative_philipson(change_var, change_vals, mode=None, base_price=36, max_subsidy=16, 
                          use_resampling=False, local_params=deepcopy(exported_parameters),
                          other_changes=None, alt_plot=None):
    # change previous parameters
    subsidies = np.linspace(0,max_subsidy,2000)
    base_value = local_params[change_var]

    if other_changes is not None:
        for key in other_changes.keys():
            local_params[key] = other_changes[key]

    # prepare canvas for graphs
    _, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xlabel("subsidy per patient ($)")
    ax1.set_ylabel("value per patient ($)")
    ax2.set_xlabel("subsidy per patient ($)")
    ax2.set_ylabel("fraction of patients tested")
    ax2.set_ylim([-0.05,1.05])
    
    for change_value in change_vals:
        # set up current experiment
        if mode == "*":
            local_params[change_var] = base_value*change_value
        elif mode == "+":
            local_params[change_var] = base_value+change_value
        else:
            local_params[change_var] = change_value
        
        philipson_values = []
        tested_frac = []
        public_value = []
        private_value = []
        planner_value = []
        
        for subsidy in subsidies:
            pub_val, priv_val, frac = philipson_objective_value(base_price-subsidy, local_params=deepcopy(local_params), use_resampling=use_resampling)
            philipson_values.append(pub_val+priv_val)
            public_value.append(pub_val)
            private_value.append(priv_val)
            planner_value.append(pub_val+priv_val-subsidy*frac)
            tested_frac.append(frac)
        
        #ax1.plot(subsidies, philipson_values, label="total")
        if mode == "*":
            label = f"{change_value}x baseline"
        elif mode == "+":
            label = f"baseline+{change_value}"
        else:
            label = f"{change_var}={change_value}"
        
        if alt_plot is None:
            ax1.plot(subsidies, planner_value, label=label)
        elif alt_plot == "public":
            ax1.plot(subsidies, public_value, label=label)
        elif alt_plot == "private":
            ax1.plot(subsidies, private_value, label=label)
        
        ax2.plot(subsidies, tested_frac, label=label)
        
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #other_changes = {"f_true": np.array([0.5, 0.5])}
    other_changes=None
    #comparative_philipson("variance_res", [1,0.5,0.05], mode="*", max_subsidy=15, use_resampling=False, other_changes=other_changes)
    #comparative_philipson("cost_res", [0,5,10], max_subsidy=15, mode="+", other_changes=other_changes)
    
    prev_list = [np.array([0.9, 0.1]), np.array([0.8,0.2]), np.array([0.7,0.3]), np.array([0.6,0.4]), np.array([0.5,0.5])]
    comparative_philipson("f_true", prev_list, other_changes=other_changes, max_subsidy=36, alt_plot="private")

    #comparative_philipson("cost_res", [1,1.25,1.5,1.75,3], mode="*")
