from mc_computations import *
from philipson_setup import *
from nun_setup import value_calculations, net_testing_value, find_tradeoff_points
from example_with_maximum import exported_parameters
import os
import matplotlib.pyplot as plt
from copy import deepcopy

def curly_brace(axes, start, end, height_one, height_two, parabola_width, exp_width, color="black"):
    #find points to connect to brace
    start_x, start_y = start
    end_x, end_y = end
    t_vals=np.linspace(0,1,1000)
    
    # plot first hook in braces
    start_connector_x = start_x + parabola_width
    start_connector_y = start_y + height_one
    x_vals = [(1-t)*start_x+t*start_connector_x for t in t_vals]
    a=start_connector_y/parabola_width**2
    y_vals = [a*(x-start_x)*(start_x+2*parabola_width-x) for x in x_vals]
    axes.plot(color=color)
    axes.plot(x_vals, y_vals, color=color)
    
    # plot straight segment
    start_of_sharp = start_connector_x+(end_x-start_x-2*parabola_width-2*exp_width)/2
    x_vals = [(1-t)*start_connector_x+t*start_of_sharp for t in t_vals]
    y_vals = [start_connector_y for x in x_vals]
    axes.plot(x_vals, y_vals, color=color)

    # half sharp segment
    x_vals = [(1-t)*start_of_sharp+t*(start_of_sharp+exp_width) for t in t_vals]
    a=np.log((height_one+height_two+start_y)/(height_one+start_y))/exp_width
    y_vals = [(height_one+start_y)*np.exp(a*(x-start_of_sharp)) for x in x_vals]
    axes.plot(x_vals, y_vals, color=color)

    # other half
    x_vals = [(1-t)*(start_of_sharp+exp_width)+t*(start_of_sharp+2*exp_width) for t in t_vals]
    x_vals = list(reversed(x_vals))
    a=np.log((height_one+start_y)/(height_one+height_two+start_y))/exp_width
    y_vals = [(height_one+height_two+start_y)*np.exp(a*(x-start_of_sharp-exp_width)) for x in x_vals]
    axes.plot(x_vals, y_vals, color=color)
    
    # other straight
    start_of_sharp = start_connector_x+(end_x-start_x-2*parabola_width-2*exp_width)/2
    end_second_straight = start_of_sharp+2*exp_width + (end_x-start_x-2*parabola_width-2*exp_width)/2
    x_vals = [(1-t)*(start_of_sharp+2*exp_width)+t*end_second_straight for t in t_vals]
    y_vals = [start_connector_y for x in x_vals]
    axes.plot(x_vals, y_vals, color=color)

    # final parabola
    x_vals = [(1-t)*end_second_straight+t*end_x for t in t_vals]
    a=start_connector_y/parabola_width**2
    y_vals = [a*(x-end_x)*(end_x-2*parabola_width-x) for x in x_vals]
    axes.plot(color=color)
    axes.plot(x_vals, y_vals, color=color)

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

def bar_philipson(base_price, change_var, change_vals, mode=None, 
                          use_resampling=False, local_params=deepcopy(exported_parameters),
                          other_changes=None, alt_plot=None):
    # change previous parameters
    base_value = local_params[change_var]

    if other_changes is not None:
        for key in other_changes.keys():
            local_params[key] = other_changes[key]

    # prepare canvas for graphs
    _, axs = plt.subplots(2,1)
  
    axs[0].set_xticks([-0.5, 0.5], labels=["value to non-testers", "value to testers"])
    axs[0].set_ylabel("value per patient ($)")
    axs[1].set_ylabel("density of patients")
    axs[1].set_xlabel("cost of empirical treatment")

    distribution_parameters = []
    label_list = []
    # divvy up the range for both non testing and testing between the different scenarios, some logic based on even or odd cases
    num_changes = len(change_vals)
    separation = 0.4/num_changes
    if num_changes % 2 == 1:
        offset = 0
    else:
        offset = 0.5
    
    lattice = [i+offset for i in range(-int(num_changes//2), 0)]
    if num_changes % 2 == 1:
        lattice.append(0)
    lattice.extend([i-offset for i in range(1, 1+int(num_changes//2))])
    bar_locations = -0.5+separation*np.array(lattice)
    
    for i,change_value in enumerate(change_vals):
        # set up current experiment
        if mode == "*":
            local_params[change_var] = base_value*change_value
        elif mode == "+":
            local_params[change_var] = base_value+change_value
        else:
            local_params[change_var] = change_value
        
        pub_val, priv_val, _ = philipson_objective_value(base_price, local_params=deepcopy(local_params), use_resampling=use_resampling)
        
        #axs[0][0].plot(subsidies, philipson_values, label="total")
        if mode == "*":
            label = f"{change_value}x baseline {change_var}"
        elif mode == "+":
            label = f"baseline {change_var}+{change_value}"
        else:
            label = f"{change_var}={change_value}"

        label_list.append(label)
        
        axs[0].bar([bar_locations[i], 1+bar_locations[i]], [pub_val, priv_val], width=separation, label=label_list[i])

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
        right_of_price = np.linspace(base_price, xmax, 1000)
        
        axs[1].plot(full_domain, gaussian(full_domain), label=label_list[i])
        axs[1].fill_between(right_of_price, gaussian(right_of_price), alpha=0.3)
        
        if i<len(distribution_parameters)-1:
            axs[1].vlines(base_price, 0, gaussian(base_price), linestyles='dashed', color="black")
        else:
            axs[1].vlines(base_price, 0, gaussian(base_price), linestyles='dashed', color="black", label="diagnostic cost")
        
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def nun_philipson(base_price, density_function, subsidy_vals=None, biased_update=False):
    # automatically set a range for subsidies if not provided
    if subsidy_vals is None:
        subsidy_vals = np.linspace(0, base_price, 100)
    
    # prepare canvas for graphs
    fig, axs = plt.subplots(2,2)
  
    # first graph is the plot of the total value for all possible subsidies
    planner_values = []
    for subsidy in subsidy_vals:
        test_value, non_test_value, testing_frac = value_calculations(base_price-subsidy, density_function, biased_update=biased_update)
        total_value = test_value + non_test_value - testing_frac*subsidy
        planner_values.append(total_value)

    # optimal subsidy calculations
    max_index = np.argmax(np.array(planner_values))
    axs[0,0].plot(subsidy_vals, planner_values)
    axs[0,0].scatter(subsidy_vals[max_index], planner_values[max_index], color="red", label="optimal subsidy")
    axs[0,0].set_xlabel("Subsidy per test ($)")
    axs[0,0].set_ylabel("Value/patient - planner costs/patient ($)")
    axs[0,0].legend()

    # private value incentives
    t_vals = np.linspace(0,1,1000)
    net_testing_vals = [net_testing_value(t) for t in t_vals]
    first_crossing, last_crossing = find_tradeoff_points(base_price)
    subs_first_crossing, subs_last_crossing = find_tradeoff_points(base_price-subsidy_vals[max_index])

    axs[0,1].plot(t_vals, net_testing_vals)
    # unsubsidized
    axs[0,1].hlines(base_price, 0, last_crossing+0.1, color="black")
    axs[0,1].text(last_crossing+0.11, base_price, "Cost of testing", fontsize=8, verticalalignment='center')
    axs[0,1].vlines(first_crossing, 0, base_price, color="black", linestyles="dashed")
    axs[0,1].vlines(last_crossing, 0, base_price, color="black", linestyles="dashed")
    #subsidized
    axs[0,1].hlines(base_price-subsidy_vals[max_index], 0, subs_last_crossing, color="black")
    axs[0,1].text(subs_last_crossing+0.01, base_price-subsidy_vals[max_index], "Cost subs. testing", fontsize=8, verticalalignment='center')
    axs[0,1].vlines(subs_first_crossing, 0, base_price-subsidy_vals[max_index], color="black", linestyles="dashed")
    axs[0,1].vlines(subs_last_crossing, 0, base_price-subsidy_vals[max_index], color="black", linestyles="dashed")

    axs[0,1].set_ylim(0, max(net_testing_vals)+1)
    axs[0,1].set_xlim(0,1)
    axs[0,1].set_ylabel("Net benefit of testing ($)")
    axs[0,1].set_xlabel("Patient risk for disease one")
    subscript = "\u209B"
    axs[0,1].set_xticks([0,first_crossing, last_crossing, subs_first_crossing, subs_last_crossing, 1], labels=[0,"L", "U", f"{"L"}{subscript}", f"{"U"}{subscript}", 1])
    fig.text(0.7, 0.63, f"Privately chooses testing", wrap=True, horizontalalignment='center', fontsize=9)
    curly_brace(axs[0,1], [first_crossing,0], [last_crossing,0], 1, 1, 0.1, 0.015, color="black")

    fill_t_vals = np.linspace(first_crossing, last_crossing ,1000)
    density_values = [density_function(t) for t in t_vals]
    axs[1,1].plot(t_vals, density_values)
    axs[1,1].set_xlim(0,1)
    axs[1,1].set_xticks([0,first_crossing, last_crossing, subs_first_crossing, subs_last_crossing, 1], labels=[0,"L", "U", f"{"L"}{subscript}", f"{"U"}{subscript}", 1])
    axs[1,1].set_xlabel("Patient risk for disease one")
    axs[1,1].set_ylabel("Number of patients")
    axs[1,1].set_yticks(())
    # unsubsidized
    axs[1,1].vlines(first_crossing, 0, density_function(first_crossing), color="black", linestyles="dashed")
    axs[1,1].vlines(last_crossing, 0, density_function(last_crossing), color="black", linestyles="dashed")
    axs[1,1].fill_between(fill_t_vals, [density_function(t) for t in fill_t_vals], alpha=0.3, color="blue")
    midpoint = (last_crossing+first_crossing)/2
    axs[1,1].text(midpoint, density_function(midpoint)/2, f"Population \ndemand for testing", horizontalalignment='center', fontsize=9)
    # subsidized
    subs_fill_t_vals = np.linspace(subs_first_crossing, subs_last_crossing, 1000)
    axs[1,1].vlines(subs_first_crossing, 0, density_function(subs_first_crossing), color="black", linestyles="dashed")
    axs[1,1].vlines(subs_last_crossing, 0, density_function(subs_last_crossing), color="black", linestyles="dashed")
    axs[1,1].fill_between(subs_fill_t_vals, [density_function(t) for t in subs_fill_t_vals], alpha=0.3, color="blue")
    axs[1,1].set_ylim(0,1.1*max(density_values))

    test_value, non_test_value, testing_frac = value_calculations(base_price-subsidy_vals[max_index], density_function)
    bar_locs = [-0.4, 0.0, 0.4]
    
    axs[1,0].set_xticks(bar_locs, labels=["Value to testers", "Value to non-testers", "Cost to planner"])
    axs[1,0].set_ylabel("Value/patient ($)")
    axs[1,0].bar(bar_locs[0:-1], [test_value, non_test_value], width=0.2)
    axs[1,0].bar([0.4], [subsidy_vals[max_index]*testing_frac], width=0.2, color="red")


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
    base_price = 20
    def edge_cases(t, edge_width=0.1, edge_height=0.1):
        # chosen so that the integral of this function on [0,1] is equal to 1
        mid_height = (1-2*edge_width*edge_height)/(1-2*edge_width)

        if t <= edge_width or t >= 1-edge_width:
            return edge_height
        else:
            return mid_height

    non_uniform = lambda t: edge_cases(t, edge_width=0.1, edge_height=0)
    uniform = lambda t: 1
    non_uniform = lambda t: 6*t*(1-t)

    def triangular(t):
        if 0<=t<=0.5:
            return 4*t
        else:
            return 4*(1-t)

    exported_parameters["cost_res"] *= 1.5
    nun_philipson(base_price, non_uniform)
    
    
    
    
    
    
    
    
    
    
    
    
    #other_changes = {"f_true": np.array([0.5, 0.5])}
    # other_changes=None
    #bar_philipson(19, "variance_res", [0.5, 1, 2], mode="*")
    # comparative_philipson("variance_res", [1,0.5,0.05], mode="*", max_subsidy=36, use_resampling=False, other_changes=other_changes)
    #bar_philipson(21, "cost_res", [0,5,10], mode="+", other_changes=other_changes)
    #comparative_philipson("m", [4,8,16,32,64], max_subsidy=36)
    
    #prev_list = [np.array([0.9, 0.1]), np.array([0.8,0.2]), np.array([0.7,0.3]), np.array([0.6,0.4]), np.array([0.5,0.5])]
    # prev_list = [np.array([0.9, 0.1])]
    # bar_philipson(19, "f_true", prev_list)

    #comparative_philipson("cost_res", [1,1.25,1.5,1.75,3], mode="*")
    
    # other_changes = {"cost_res": 1*exported_parameters["cost_res"], "m": 1}
    # delta=0.05
    # prev_list=[np.array([0.9-delta*i, 0.1+delta*i]) for i in range(0, 7)]
    # philipson_one_subsidy_experiment(prev_list=prev_list, other_changes=other_changes)
