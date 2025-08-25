import numpy as np
from parameters import *

rng = np.random.default_rng(42069)  # the generator used for our experiments, can set a seed for consistency

def h_helper(f):
    h=-drug_costs  # holds values of treatments, begin by subtracting costs
    for i in range(0, num_treatments):
        for j in range(0, num_pathogens):
            r_ij = resistance[i,j]
            h[i] += f[j]*(r_ij*(cost_per_qaly*qaly_if_res-cost_res)+(1-r_ij)*cost_per_qaly*qaly_if_susc)
    
    return np.array(h)

# uses the above helper
true_scores = h_helper(f_true)

# returns the argument of the best empirical diagnosis given that the believed distribution is f_realized
def argmax_helper(f_realized):
    treatment_scores = h_helper(f_realized)
    return np.argmax(treatment_scores)

def objective_function(p_test):
    prior_alpha = m*f_0  # retrieve information on prior
    emp_net_benefit_estimate = 0

    # perform Monte-Carlo sampling
    for _ in range(0, diagnostic_realizations):
        # one run of testing p_test of our patients to improve empirical diagnoses
        other_diagnostics = rng.multinomial((num_patients-1)*p_test, f_true)
        prob_estimates = np.zeros(num_treatments)  # will hold probability of treatment being best
        for _ in range(0, prior_update_realizations):
            # update prior based on the results of our randomly generated measurements
            posterior_alpha = prior_alpha + other_diagnostics
            f_realized = rng.dirichlet(posterior_alpha)
            prob_estimates[argmax_helper(f_realized)] += 1
        
        prob_estimates = prob_estimates/prior_update_realizations  # normalize results
        emp_net_benefit_estimate += np.dot(prob_estimates, true_scores)
    
    emp_net_benefit_estimate = emp_net_benefit_estimate/diagnostic_realizations
    testing_net_benefit = np.dot(f_true, optimal_net_benefits)-cost_test

    obj_value = (1-(1-p_ignore)*p_test)*emp_net_benefit_estimate + (1-p_ignore)*p_test*testing_net_benefit - p_test*p_ignore*cost_test

    return obj_value

if __name__ == "__main__":
    print(objective_function(0.5))