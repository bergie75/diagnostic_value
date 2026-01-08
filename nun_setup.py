# math and user-defined
import numpy as np
#from math import floor, ceil
from scipy.special import erf
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt
from example_with_maximum import exported_parameters
# logistical
# import os
from copy import deepcopy
# import itertools
# import re
# import string
# plotting
# from plotting_pipelines import philipson_plots

rng = np.random.default_rng()

# based on the planner's current uncertainty, how likely is each diagnosis?
def planner_decision_probs(alpha):
    # alpha is the vector parametrizing the planner's prior
    #prob_disease_2 = betainc(alpha[0], alpha[1], 1/2)
    #prob_disease_1 = 1 - prob_disease_2

    prob_disease_1 = alpha[0]/np.sum(alpha)
    prob_disease_2 = alpha[1]/np.sum(alpha)

    return np.array([prob_disease_1, prob_disease_2])

def bias_array(bias_value):
    if 0 <= bias_value <= 1/2:
        return np.array([2*bias_value, 0])
    else:
        return np.array([1,2*bias_value-1])

def net_testing_value(bias_value, local_params=exported_parameters):
    # unpack important parameters
    f_true = local_params["f_true"]
    cost_per_qaly = local_params["cost_per_qaly"]
    qaly_if_susc = local_params["qaly_if_susc"]
    qaly_if_res = local_params["qaly_if_res"]
    drug_costs = local_params["drug_costs"]
    cost_res = local_params["cost_res"]
    f_0 = local_params["f_0"]
    
    # modify the planner's beliefs based on the patient at hand
    bias_arr = bias_array(bias_value)
    planner_prob_disease_one = np.dot(f_0, bias_arr)
    planner_probs = [planner_prob_disease_one, 1-planner_prob_disease_one]

    # modify the objective probabilities based on the patient at hand
    true_prob_disease_one = np.dot(f_true, bias_arr)
    true_probs = [true_prob_disease_one, 1-true_prob_disease_one]

    net_testing_value = (true_probs[0]*(planner_probs[0]-1)*(drug_costs[0]-cost_per_qaly*qaly_if_susc[0])
    +true_probs[0]*planner_probs[1]*(drug_costs[1]+cost_res[0]-cost_per_qaly*qaly_if_res[0])
    +true_probs[1]*planner_probs[0]*(drug_costs[0]+cost_res[1]-cost_per_qaly*qaly_if_res[1])
    +true_probs[1]*(planner_probs[1]-1)*(drug_costs[1]-cost_per_qaly*qaly_if_susc[1]))

    return net_testing_value

def empirical_value(bias_value, local_params=exported_parameters):
    # unpack important parameters
    f_true = local_params["f_true"]
    cost_per_qaly = local_params["cost_per_qaly"]
    qaly_if_susc = local_params["qaly_if_susc"]
    qaly_if_res = local_params["qaly_if_res"]
    drug_costs = local_params["drug_costs"]
    cost_res = local_params["cost_res"]
    f_0 = local_params["f_0"]
    
    # modify the planner's beliefs based on the patient at hand
    bias_arr = bias_array(bias_value)
    planner_prob_disease_one = np.dot(f_0, bias_arr)
    planner_probs = [planner_prob_disease_one, 1-planner_prob_disease_one]

    # modify the objective probabilities based on the patient at hand
    true_prob_disease_one = np.dot(f_true, bias_arr)
    true_probs = [true_prob_disease_one, 1-true_prob_disease_one]

    empirical_value = (true_probs[0]*(planner_probs[0])*(-drug_costs[0]+cost_per_qaly*qaly_if_susc[0])
    +true_probs[0]*planner_probs[1]*(-drug_costs[1]-cost_res[0]+cost_per_qaly*qaly_if_res[0])
    +true_probs[1]*planner_probs[0]*(-drug_costs[0]-cost_res[1]+cost_per_qaly*qaly_if_res[1])
    +true_probs[1]*(planner_probs[1])*(-drug_costs[1]-cost_per_qaly*qaly_if_susc[1]))

    return empirical_value

def testing_value(bias_value, local_params=exported_parameters):
    # unpack important parameters
    f_true = local_params["f_true"]
    cost_per_qaly = local_params["cost_per_qaly"]
    qaly_if_susc = local_params["qaly_if_susc"]
    drug_costs = local_params["drug_costs"]
    
    # modify the planner's beliefs based on the patient at hand
    bias_arr = bias_array(bias_value)

    # modify the objective probabilities based on the patient at hand
    true_prob_disease_one = np.dot(f_true, bias_arr)
    true_probs = [true_prob_disease_one, 1-true_prob_disease_one]

    testing_value = (true_probs[0]*(-drug_costs[0]+cost_per_qaly*qaly_if_susc[0])
                     +true_probs[1]*(-drug_costs[1]+cost_per_qaly*qaly_if_susc[1]))

    return testing_value

# density function must take a bias value and return the probability density function at that value
def value_calculations(base_price, density_function, local_params=exported_parameters):
    # separate dictionaries to pass as arguments when computing average values
    non_testing_params = deepcopy(local_params)
    testing_params = deepcopy(local_params)
    
    # unpack values used below to modify parameters
    m = testing_params["m"]
    f_0 = testing_params["f_0"]
    f_true = testing_params["f_true"]
    num_patients = testing_params["num_patients"]

    # first, find the bias values that bracket the region where testing should occur
    t_vals = np.linspace(0,1,1000)
    net_testing_vals = [net_testing_value(t, local_params=local_params) for t in t_vals]

    # find crossing values
    for i,test_val in enumerate(net_testing_vals):
        if test_val >= base_price:
            first_index = i
            break
    
    for i in range(first_index, len(net_testing_vals)):
        test_val = net_testing_vals[i]
        if test_val < base_price or i==len(net_testing_vals)-1:
            last_index = i
            break

    # compute the fraction of patients who will undergo diagnostic testing
    testing_frac = quad(lambda t: density_function(t), t_vals[first_index], t_vals[last_index])[0]  # second entry is error

    # compute the value to testers
    testing_integrand = lambda t: density_function(t)*testing_value(t, local_params=testing_params)
    test_value = quad(testing_integrand, t_vals[first_index], t_vals[last_index])[0]-base_price*testing_frac
    
    # update with measurement results (assume to be average, for simplicity)
    updated_m = m + num_patients*testing_frac
    updated_f_0 = (m*f_0 + num_patients*testing_frac*f_true)/updated_m
    
    # use the results of the testing to update physician beliefs for non-testers
    non_testing_params["m"] = updated_m
    non_testing_params["f_0"] = updated_f_0
    non_testing_integrand = lambda t: density_function(t)*empirical_value(t, local_params=non_testing_params)
    non_test_value = quad(non_testing_integrand, 0, t_vals[first_index])[0] + quad(non_testing_integrand, t_vals[last_index], 1)[0]

    return test_value, non_test_value, testing_frac

if __name__ == "__main__":
    base_price=19
    num_patients = exported_parameters["num_patients"]
    
    import matplotlib.pyplot as plt
    t_vals = np.linspace(0,1,1000)
    net_testing_vals = [net_testing_value(t) for t in t_vals]

    # find crossing values
    for i,test_val in enumerate(net_testing_vals):
        if test_val >= base_price:
            first_index = i
            break
    
    for i in range(first_index, len(net_testing_vals)):
        test_val = net_testing_vals[i]
        if test_val <= base_price:
            last_index = i
            break

    # two possible risk distributions
    uniform = lambda t: 1

    def edge_cases(t, edge_width=0.1, edge_height=0.1):
        # chosen so that the integral of this function on [0,1] is equal to 1
        mid_height = (1-2*edge_width*edge_height)/(1-2*edge_width)

        if t <= edge_width or t >= 1-edge_width:
            return edge_height
        else:
            return mid_height

    non_uniform = lambda t: edge_cases(t, edge_width=0.3, edge_height=0.1)

    plt.plot(t_vals, net_testing_vals)
    plt.hlines(base_price, 0, t_vals[last_index]+0.1, color="black")
    plt.text(t_vals[last_index]+0.11, base_price, "Cost of testing", fontsize=8, verticalalignment='center')
    plt.vlines(t_vals[first_index], 0, base_price, color="black", linestyles="dashed")
    plt.vlines(t_vals[last_index], 0, base_price, color="black", linestyles="dashed")
    plt.ylim(0, max(net_testing_vals)+1)
    plt.xlim(0,1)
    plt.ylabel("Net benefit of testing ($)")
    plt.xlabel("Patient risk for disease one")
    plt.xticks([0,t_vals[first_index], t_vals[last_index], 1], labels=[0,"L", "U", 1])
    plt.figtext(
    0.43, 0.07, f"Privately chooses testing", wrap=True, horizontalalignment='center', fontsize=8)
    print(last_index)
    plt.show()

    pop_density = lambda t: edge_cases(t, edge_width=0.3)
    density_values = [pop_density(t) for t in t_vals]
    plt.plot(t_vals, density_values)
    plt.xlim(0,1)
    plt.xticks([0,t_vals[first_index], t_vals[last_index], 1], labels=[0,"L", "U", 1])
    plt.xlabel("Patient risk for disease one")
    plt.ylabel("Number of patients")
    plt.yticks(())
    plt.vlines(t_vals[first_index], 0, density_values[first_index], color="black", linestyles="dashed")
    plt.vlines(t_vals[last_index], 0, density_values[last_index], color="black", linestyles="dashed")
    plt.fill_between(t_vals[first_index:last_index+1], density_values[first_index:last_index+1], alpha=0.3)
    plt.figtext(0.465, 0.47, f"Population \ndemand for testing", wrap=True, horizontalalignment='center', fontsize=9)
    plt.ylim(0,1.1*max(density_values))
    plt.show()

    test_value, non_test_value, testing_frac = value_calculations(base_price, pop_density)
    bar_locs = [-0.2, 0.2]
    
    plt.xticks(bar_locs, labels=["Value to testers", "Value to non-testers"])
    plt.ylabel("Value per patient ($)")
    plt.bar(bar_locs, [test_value, non_test_value], width=0.2)
    plt.show()


