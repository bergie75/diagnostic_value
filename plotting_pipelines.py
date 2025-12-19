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
    _, axs = plt.subplots(2,2)

    
    axs[0][0].set_xlabel("subsidy per patient ($)")
    axs[0][0].set_ylabel("value per patient ($)")
    axs[1][0].set_xlabel("subsidy per patient ($)")
    axs[1][0].set_ylabel("fraction of patients tested")
    axs[1][1].set_ylabel("density of patients")
    axs[1][1].set_xlabel("cost of empirical treatment with optimal subsidy")
    axs[0][1].remove()
    
    # so whole curve of frac testing can be seen
    axs[1][0].set_ylim([-0.05,1.05])

    distribution_parameters = []
    label_list = []
    
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
        
        max_index = np.argmax(np.array(planner_value))
        
        #axs[0][0].plot(subsidies, philipson_values, label="total")
        if mode == "*":
            label = f"{change_value}x baseline {change_var}"
        elif mode == "+":
            label = f"baseline {change_var}+{change_value}"
        else:
            label = f"{change_var}={change_value}"

        label_list.append(label)
        
        if alt_plot is None:
            axs[0][0].plot(subsidies, planner_value, label=label, zorder=0)
        elif alt_plot == "public":
            axs[0][0].plot(subsidies, public_value, label=label, zorder=0)
        elif alt_plot == "private":
            axs[0][0].plot(subsidies, private_value, label=label, zorder=0)
        
        axs[0][0].scatter(subsidies[max_index], planner_value[max_index], color="black", zorder=10, marker="x")
        
        axs[1][0].plot(subsidies, tested_frac, label=label)
        axs[1][0].scatter(subsidies[max_index], tested_frac[max_index], color="black", zorder=10, marker="x")

        # prepare a Philipson plot for the number of patients testing
        res_cost_mean = local_params["cost_res"]
        res_cost_variance = local_params["variance_res"]

        mu, sigma_squared = final_mean_and_variance(res_cost_mean, res_cost_variance, local_params=deepcopy(local_params))
        sigma = np.sqrt(sigma_squared)
        distribution_parameters.append([mu, sigma])
        
    xmax = max([x[0]+3*x[1] for x in distribution_parameters])
    xmin = min([x[0]-3*x[1] for x in distribution_parameters])
    full_domain = np.linspace(xmin, xmax, 1000)
    
    for i,dist_pars in enumerate(distribution_parameters):
        mu, sigma = dist_pars
        gaussian = lambda x: 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
        right_of_price = np.linspace(base_price-subsidies[max_index], xmax, 1000)
        axs[1][1].vlines(base_price-subsidies[max_index], 0, gaussian(base_price-subsidies[max_index]), linestyles='dashed', color="black")
        axs[1][1].plot(full_domain, gaussian(full_domain), label=label_list[i])
        axs[1][1].fill_between(right_of_price, gaussian(right_of_price), alpha=0.3)
        
    axs[0][0].legend()
    axs[1][0].legend()
    axs[1][1].legend()
    plt.tight_layout()
    plt.show()

def philipson_temporal_experiment(prev_list, base_price=36, max_subsidy=None, local_params=deepcopy(exported_parameters),
                                  other_changes=None, use_resampling=False):
    # change previous parameters
    if max_subsidy is None:
        max_subsidy=base_price
    subsidies = np.linspace(0,max_subsidy,2000)

    if other_changes is not None:
        for key in other_changes.keys():
            local_params[key] = other_changes[key]

    num_patients = local_params["num_patients"]
    m = local_params["m"]
    f_true = local_params["f_true"]
    
    # prepare canvas for graphs
    _, axs = plt.subplots(2,2)
    axs[0][0].set_xlabel("experiment number")
    axs[0][0].set_ylabel("value per patient ($)")
    axs[1][0].set_xlabel("experiment number")
    axs[1][0].set_ylabel("optimal subsidy ($)")
    axs[1][1].set_xlabel("experiment number")
    axs[1][1].set_ylabel("optimal testing fraction")
    
    for i,disease_prevalence in enumerate(prev_list):
        local_params["f_true"] = disease_prevalence
        
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
        
        max_index = np.argmax(planner_value)
        
        axs[0][0].scatter(i+1, planner_value[max_index], color="red")
        axs[1][0].scatter(i+1, subsidies[max_index], color="red")
        optimal_test_frac = tested_frac[max_index]
        axs[1][1].scatter(i+1, optimal_test_frac, color="red")

        # update prior
        prior_alpha = m*local_params["f_0"]
        local_params["f_0"] = planner_decision_probs(prior_alpha+num_patients*f_true*optimal_test_frac)
        
    plt.tight_layout()
    plt.show()

def philipson_one_subsidy_experiment(prev_list, base_price=36, max_subsidy=None, local_params=deepcopy(exported_parameters),
                                  other_changes=None, use_resampling=False):
    # change previous parameters
    if max_subsidy is None:
        max_subsidy=base_price
    subsidies = np.linspace(0,max_subsidy,2000)

    if other_changes is not None:
        for key in other_changes.keys():
            local_params[key] = other_changes[key]

    num_patients = local_params["num_patients"]
    m = local_params["m"]
    f_true = local_params["f_true"]
    base_f_true = local_params["f_true"]
    base_f_0 = local_params["f_0"]
    
    tested_frac = [[] for _ in range(0,2001)]
    planner_value = np.zeros(2001)
    
    for subsidy_index, subsidy in enumerate(subsidies):
        local_params["f_true"] = base_f_true
        local_params["f_0"] = base_f_0
        
        for disease_prevalence in prev_list:
            local_params["f_true"] = disease_prevalence
            pub_val, priv_val, frac = philipson_objective_value(base_price-subsidy, local_params=deepcopy(local_params), use_resampling=use_resampling)
            planner_value[subsidy_index] += pub_val+priv_val-subsidy*frac
            tested_frac[subsidy_index].append(frac)

            # update prior
            prior_alpha = m*local_params["f_0"]
            local_params["f_0"] = planner_decision_probs(prior_alpha+num_patients*f_true*frac)
        
    planner_value /= len(prev_list)  # so values can be interpreted as an average
    max_index = np.argmax(planner_value)

    print(f"Optimal subsidy: ${subsidies[max_index]:.2f}")
    print(f"Optimal value: ${planner_value[max_index]:.2f}")
    plt.plot(list(range(0, 7)), tested_frac[max_index], marker="o")
    plt.xlabel("Time index")
    plt.ylabel("Fraction tested")
    plt.show()

if __name__ == "__main__":
    #other_changes = {"f_true": np.array([0.5, 0.5])}
    other_changes=None
    comparative_philipson("variance_res", [1,0.5,0.05], mode="*", max_subsidy=36, use_resampling=False, other_changes=other_changes)
    comparative_philipson("variance_res", [1,0.5,0.05], mode="*", max_subsidy=36, use_resampling=False, other_changes=other_changes)
    #comparative_philipson("cost_res", [0,5,10], max_subsidy=36, mode="+", other_changes=other_changes)
    #comparative_philipson("m", [4,8,16,32,64], max_subsidy=36)
    
    prev_list = [np.array([0.9, 0.1]), np.array([0.8,0.2]), np.array([0.7,0.3]), np.array([0.6,0.4]), np.array([0.5,0.5])]
    #comparative_philipson("f_0", prev_list, other_changes=other_changes, max_subsidy=36)

    #comparative_philipson("cost_res", [1,1.25,1.5,1.75,3], mode="*")
    
    # other_changes = {"cost_res": 1*exported_parameters["cost_res"], "m": 1}
    # delta=0.05
    # prev_list=[np.array([0.9-delta*i, 0.1+delta*i]) for i in range(0, 7)]
    # philipson_one_subsidy_experiment(prev_list=prev_list, other_changes=other_changes)
