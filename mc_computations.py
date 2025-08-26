import numpy as np
import time
import matplotlib.pyplot as plt
from parameters import *

rng = np.random.default_rng()  # the generator used for our experiments, can set a seed for consistency

def h_helper(f):
    return -drug_costs + np.matmul(res_benefit_matrix, f)  # holds values of treatments, begin by subtracting costs

# uses the above helper
true_scores = h_helper(f_true)

# returns the argument of the best empirical diagnosis given that the believed distribution is f_realized
def argmax_helper(f_realized):
    return np.argmax(h_helper(f_realized))

def objective_function(p_test):
    prior_alpha = m*f_0  # retrieve information on prior
    emp_net_benefit_estimate = 0

    # perform Monte-Carlo sampling
    for _ in range(0, diagnostic_realizations):
        # one run of testing p_test of our patients to improve empirical diagnoses
        other_diagnostics = rng.multinomial((num_patients-1)*p_test, f_true)
        prob_estimates = np.zeros(num_treatments)  # will hold probability of treatment being best
        # update prior based on the results of our randomly generated measurements
        posterior_alpha = prior_alpha + other_diagnostics
        for _ in range(0, prior_update_realizations):
            f_realized = rng.dirichlet(posterior_alpha)
            prob_estimates[argmax_helper(f_realized)] += 1
        
        prob_estimates = prob_estimates/prior_update_realizations  # normalize results
        emp_net_benefit_estimate += np.dot(prob_estimates, true_scores)
    
    emp_net_benefit_estimate = emp_net_benefit_estimate/diagnostic_realizations
    testing_net_benefit = np.dot(f_true, optimal_net_benefits)-cost_test
    obj_value = (1-(1-p_ignore)*p_test)*emp_net_benefit_estimate + (1-p_ignore)*p_test*testing_net_benefit - p_test*p_ignore*cost_test

    return obj_value

if __name__ == "__main__":
    p_test_vals = np.array(range(0, num_patients+1))/num_patients
    obj_fun_vals = np.array([objective_function(p_test) for p_test in p_test_vals])
    plt.plot(p_test_vals, obj_fun_vals)
    plt.title("Tradeoffs between public and private value of diagnostics")
    plt.xlabel("Fraction tested")
    plt.ylabel("Net benefit ($)")
    plt.show()

    # a small diagnostic to see how likely the prior is to pick out the correct treatment choice
    # print(true_scores)
    # count_correct = 0
    # max_index = np.argmax(true_scores)

    # for _ in range(0, prior_update_realizations):
    #     f_realized = rng.dirichlet(m*f_0)
    #     f_scores = h_helper(f_realized)
    #     count_correct += (np.argmax(f_scores) == max_index)
    
    # print(f"Probability of correct choice = {count_correct/prior_update_realizations*100}")
