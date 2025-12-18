# math and user-defined
import numpy as np
#from math import floor, ceil
from scipy.special import erf
from scipy.stats import norm
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

def final_mean_and_variance(means, variances, frac_tested=None, local_params=deepcopy(exported_parameters)):
    # unpack important parameters
    f_true = local_params["f_true"]
    cost_per_qaly = local_params["cost_per_qaly"]
    qaly_if_susc = local_params["qaly_if_susc"]
    qaly_if_res = local_params["qaly_if_res"]
    drug_costs = local_params["drug_costs"]
    
    # calculate parameters of planner prior
    m = local_params["m"]
    f_0 = local_params["f_0"]
    alpha = m*f_0

    if frac_tested is None:
        planner_probs = planner_decision_probs(alpha)
    else:
        num_patients = local_params["num_patients"]
        # when using the other form of planner probs
        unnormed = alpha+num_patients*frac_tested*f_true
        planner_probs = unnormed/np.sum(unnormed)

    new_mean = (f_true[0]*(planner_probs[0]-1)*(drug_costs[0]-cost_per_qaly*qaly_if_susc[0])
    +f_true[0]*planner_probs[1]*(drug_costs[1]+means[0]-cost_per_qaly*qaly_if_res[0])
    +f_true[1]*planner_probs[0]*(drug_costs[0]+means[1]-cost_per_qaly*qaly_if_res[1])
    +f_true[1]*(planner_probs[1]-1)*(drug_costs[1]-cost_per_qaly*qaly_if_susc[1]))

    new_variance = variances[0]*(f_true[0]*planner_probs[1])**2+variances[1]*(f_true[1]*planner_probs[0])**2

    return new_mean, new_variance

def resampled_mean(price, means, variances, frac_tested, use_resampling=False, local_params=deepcopy(exported_parameters)):
    # unpack important parameters
    f_true = local_params["f_true"]
    cost_per_qaly = local_params["cost_per_qaly"]
    qaly_if_susc = local_params["qaly_if_susc"]
    qaly_if_res = local_params["qaly_if_res"]
    drug_costs = local_params["drug_costs"]
    
    # calculate parameters of planner prior
    m = local_params["m"]
    f_0 = local_params["f_0"]
    alpha = m*f_0

    # we use these to compute the conditional mean of patients who forego a test
    unadjusted_planner_probs = planner_decision_probs(alpha)

    if use_resampling:
        # calculate coefficients for conditional mean computation
        coef_res_0 = f_true[0]*unadjusted_planner_probs[1]
        coef_res_1 = f_true[1]*unadjusted_planner_probs[0]
        density_cutoff = (price + coef_res_0*(drug_costs[0]-cost_per_qaly*qaly_if_susc[0]-drug_costs[1]+cost_per_qaly*qaly_if_res[0])
                    +coef_res_1*(drug_costs[1]-cost_per_qaly*qaly_if_susc[1]-drug_costs[0]+cost_per_qaly*qaly_if_res[1]))
        
        # the prefactor measuring the ratio of variances between the cost of resistance and the linear combination
        # found in our parent Gaussian
        variance_ratio_0 = coef_res_0*variances[0]/np.sqrt(variances[0]*coef_res_0**2+variances[1]*coef_res_1**2)
        variance_ratio_1 = coef_res_1*variances[1]/np.sqrt(variances[0]*coef_res_0**2+variances[1]*coef_res_1**2)

        # modify the formula with a ratio of pdf to cdf evaluated at z-score of the price
        z_score = (density_cutoff-coef_res_0*means[0]-coef_res_1*means[1])/np.sqrt(variances[0]*coef_res_0**2+variances[1]*coef_res_1**2)
        pdf_cdf_ratio = np.sqrt(2/np.pi)*np.exp(-1/2*z_score**2)/(1+erf(z_score))

        # the newly modified means for the cost of resistance to each disease
        resampled_mean_0 = means[0]-variance_ratio_0*pdf_cdf_ratio
        resampled_mean_1 = means[1]-variance_ratio_1*pdf_cdf_ratio
    else:
        resampled_mean_0 = means[0]
        resampled_mean_1 = means[1]

    # these are the planner probabilities we use in our final answer
    num_patients = local_params["num_patients"]
    unnormed_update = alpha+num_patients*frac_tested*f_true
    planner_probs = unnormed_update/np.sum(unnormed_update)

    new_mean = (f_true[0]*(planner_probs[0])*(-drug_costs[0]+cost_per_qaly*qaly_if_susc[0])
    +f_true[0]*planner_probs[1]*(-drug_costs[1]-resampled_mean_0+cost_per_qaly*qaly_if_res[0])
    +f_true[1]*planner_probs[0]*(-drug_costs[0]-resampled_mean_1+cost_per_qaly*qaly_if_res[1])
    +f_true[1]*(planner_probs[1])*(-drug_costs[1]+cost_per_qaly*qaly_if_susc[1]))
    
    return new_mean

def philipson_objective_value(price, forced_test_frac=None, use_resampling=False, local_params=deepcopy(exported_parameters)):
    # find important quantities for stochastic cost of resistance
    res_cost_means = local_params["cost_res"]
    res_cost_variances = local_params["variance_res"]

    # needed for value calculations
    f_true = local_params["f_true"]
    cost_per_qaly = local_params["cost_per_qaly"]
    qaly_if_susc = local_params["qaly_if_susc"]
    drug_costs = local_params["drug_costs"]

    no_other_tests_mean, no_other_tests_variance = final_mean_and_variance(res_cost_means, res_cost_variances,
                                                                            local_params=deepcopy(local_params))
    
    # what fraction should test given the relevant values, or use a hand-picked value
    if forced_test_frac is None:
        frac_to_test = 1-norm.cdf(price, no_other_tests_mean, np.sqrt(no_other_tests_variance))
    else:
        frac_to_test = forced_test_frac
    
    # private value from individuals who tested
    private_value = frac_to_test*(f_true[0]*(cost_per_qaly*qaly_if_susc[0]-drug_costs[0])+
                                  f_true[1]*(cost_per_qaly*qaly_if_susc[1]-drug_costs[1])-
                                  price)
    
    # public value for individuals who did not test
    pub_val_mean = resampled_mean(price, res_cost_means, res_cost_variances, frac_to_test, use_resampling=use_resampling, local_params=deepcopy(local_params))
    public_value = (1-frac_to_test)*pub_val_mean

    #print(f"public: {public_value}, private: {private_value}")
    return public_value, private_value, frac_to_test

if __name__ == "__main__":
    pass
    
