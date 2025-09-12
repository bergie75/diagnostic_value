# math and user-defined
import numpy as np
from scipy.special import betainc
import matplotlib.pyplot as plt
from example_with_maximum import exported_parameters, par_indices
# logistical
import os
import itertools

# the generator used for our experiments, can set a seed for consistency
rng = np.random.default_rng()

def res_ben_matrix(parameters=exported_parameters):
    num_treatments = parameters[par_indices["num_treatments"]]
    num_pathogens = parameters[par_indices["num_pathogens"]]
    resistance = parameters[par_indices["resistance"]]
    cost_per_qaly = parameters[par_indices["cost_per_qaly"]]
    qaly_if_res = parameters[par_indices["qaly_if_res"]]
    qaly_if_susc = parameters[par_indices["qaly_if_susc"]]
    cost_res = parameters[par_indices["cost_res"]]

    res_benefit_matrix = np.zeros((num_treatments, num_pathogens))
    for i in range(0, num_treatments):
      for j in range(0, num_pathogens):
            r_ij = resistance[i,j]
            res_benefit_matrix[i,j] = resistance[i,j]*(r_ij*(cost_per_qaly*qaly_if_res[i]-cost_res[j])+(1-r_ij)*cost_per_qaly*qaly_if_susc[i])
    
    return res_benefit_matrix
            
def opt_net_benefits(res_benefit_matrix, parameters=exported_parameters):
    optimal_net_benefits_list = []
    num_pathogens = parameters[par_indices["num_pathogens"]]

    for j in range(0, num_pathogens):
        benefits = res_benefit_matrix[:,j]
        optimal_net_benefits_list.append(np.max(benefits))

    return np.array(optimal_net_benefits_list)

def h_helper(f, parameters=exported_parameters):
    drug_costs = parameters[par_indices["drug_costs"]]
    res_benefit_matrix = res_ben_matrix(parameters)
    return -drug_costs + np.matmul(res_benefit_matrix, f)  # holds values of treatments, begin by subtracting costs

# returns the argument of the best empirical diagnosis given that the believed distribution is f_realized
def argmax_helper(f_realized):
    return np.argmax(h_helper(f_realized))

def objective_function_for_2_by_2(p_test, parameters=exported_parameters, print_update=False):
    # unpack needed values from parameter list
    drug_costs = parameters[par_indices["drug_costs"]]
    cost_test = parameters[par_indices["cost_test"]]
    diagnostic_realizations = parameters[par_indices["diagnostic_realizations"]]
    num_patients = parameters[par_indices["num_patients"]]
    num_treatments = parameters[par_indices["num_treatments"]]
    f_true = parameters[par_indices["f_true"]]
    prior_alpha = parameters[par_indices["m"]]*parameters[par_indices["f_0"]]
    p_ignore = parameters[par_indices["p_ignore"]]
    f_true = parameters[par_indices["f_true"]]
    
    # compute useful intermediate quantities with parameter list
    res_benefit_matrix = res_ben_matrix(parameters)
    optimal_net_benefits = opt_net_benefits(res_benefit_matrix, parameters)
    true_scores = h_helper(f_true)
    
    # an accumulator for the empirical treatment part of the objective function
    emp_net_benefit_estimate = 0

    # compute thresholds for comparisons
    thresh_num = drug_costs[0]-drug_costs[1]+res_benefit_matrix[1,1]-res_benefit_matrix[0,1]
    thresh_denom = res_benefit_matrix[0,0]+res_benefit_matrix[1,1]-res_benefit_matrix[0,1]-res_benefit_matrix[1,0]
    threshold = thresh_num/thresh_denom

    # perform Monte-Carlo sampling
    for _ in range(0, diagnostic_realizations):
        # one run of testing p_test of our patients to improve empirical diagnoses
        other_diagnostics = rng.multinomial((num_patients-1)*p_test, f_true)
        prob_estimates = np.zeros(num_treatments)  # will hold probability of treatment being best
        # update prior based on the results of our randomly generated measurements
        posterior_alpha = prior_alpha + other_diagnostics

        # use theoretical results to skip inner loop
        prob_estimates[1] = betainc(posterior_alpha[0],posterior_alpha[1],threshold)
        prob_estimates[0] = 1-prob_estimates[1]
        
        emp_net_benefit_estimate += np.dot(prob_estimates, true_scores)
    
    emp_net_benefit_estimate = emp_net_benefit_estimate/diagnostic_realizations
    testing_net_benefit = np.dot(f_true, optimal_net_benefits)-cost_test
    obj_value = (1-(1-p_ignore)*p_test)*emp_net_benefit_estimate + (1-p_ignore)*p_test*testing_net_benefit - p_test*p_ignore*cost_test

    if print_update:
        print(f"Completed calculation for {p_test}")

    return obj_value

def find_critical_cost(parameters=exported_parameters):
    f_true = parameters[par_indices["f_true"]]
    res_benefit_matrix = res_ben_matrix(parameters=parameters)
    optimal_net_benefits = opt_net_benefits(res_benefit_matrix, parameters=exported_parameters)
    
    return np.dot(f_true, optimal_net_benefits)

def run_experiment(experiment_name, vars_changed, changes_to_try, local_params=exported_parameters):
    # preliminaries to save the results
    cwd = os.getcwd()
    save_to = os.path.join(cwd,"diagnostic_value","longer_runs", experiment_name)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # generate testing frequencies
    num_patients = exported_parameters[par_indices["num_patients"]]  # a constant from our configuration file
    p_test_vals = np.array(range(0, num_patients+1))/num_patients

    # generate tests
    combinations = list(itertools.product(*changes_to_try))
    
    for combo in combinations:
        # prepare to save results to specific file and change local variables for experiment
        print(f"Working on combination={combo}")
        label = ""
        
        for i,var_name in enumerate(vars_changed):
            label += f"{var_name}_{combo[i]}_"
            local_params[par_indices[var_name]] = combo[i]
        
        label = label[:-1]  # strip trailing underscore

        # carry out experiment and add results to plot
        obj_fun_vals = np.array([objective_function_for_2_by_2(p_test, parameters=local_params) for p_test in p_test_vals])
        plt.plot(p_test_vals, obj_fun_vals)
        np.save(os.path.join(save_to, label), obj_fun_vals)  # save results to view later

    plt.title("Tradeoffs between public and private value of diagnostics")
    plt.xlabel("Net benefit ($)")
    plt.ylabel("Sum of public and private net benefit")
    plt.legend([f"{combo}" for combo in combinations])
    plt.show()

if __name__ == "__main__":
    vars_changed = ["cost_test", "p_ignore"]
    changes_to_try = [np.linspace(4,6,11), [1]]
    run_experiment("ignoring_tests", vars_changed, changes_to_try)
    # cwd = os.getcwd()
    # filename = "ignoring_tests"
    # save_to = os.path.join(cwd,"diagnostic_value","longer_runs", filename)

    # if not os.path.exists(save_to):
    #     os.makedirs(save_to)

    # num_patients = exported_parameters[par_indices["num_patients"]]  # a constant from our configuration file
    # p_test_vals = np.array(range(0, num_patients+1))/num_patients
    # test_cost_vals = np.linspace(2,4,11)  # we will examine these costs to see how the net benefit curve changes
    # local_params = exported_parameters
    # ignore_probs = [1]
    # # worse_res_factor = 1
    # # local_params[par_indices["qaly_if_res"]] = local_params[par_indices["qaly_if_res"]]/worse_res_factor
    # # local_params[par_indices["cost_test"]] = find_critical_cost(local_params)
    
    # for cost_test in test_cost_vals:
    #     for ignore_prob in ignore_probs:
    #         print(f"Working on test cost=${cost_test}")
    #         local_params[par_indices["p_ignore"]] = ignore_prob
    #         local_params[par_indices["cost_test"]] = cost_test
    #         obj_fun_vals = np.array([objective_function_for_2_by_2(p_test, parameters=local_params) for p_test in p_test_vals])
    #         plt.plot(p_test_vals, obj_fun_vals)
    #         plt.title("Tradeoffs between public and private value of diagnostics")
    #         plt.xlabel("Net benefit ($)")
    #         plt.ylabel("Sum of public and private net benefit")
    #         np.save(os.path.join(save_to, f"test_cost_{cost_test}_ignore_prob_{ignore_prob}"), obj_fun_vals)  # save results to view later

    # plt.legend([f"${cost_test:.2f}" for cost_test in test_cost_vals])
    # plt.show()

    # p_test_vals = np.array(range(0, num_patients+1))/num_patients
    # evsi_vals = np.array([evsi(p_test) for p_test in p_test_vals])
    # plt.plot(p_test_vals, evsi_vals)
    # plt.title("EVSI of testing")
    # plt.xlabel("Fraction tested")
    # plt.ylabel("EVSI ($)")
    # plt.show()

    # a small diagnostic to see how likely the prior is to pick out the correct treatment choice
    # print(true_scores)
    # count_correct = 0
    # max_index = np.argmax(true_scores)

    # for _ in range(0, prior_update_realizations):
    #     f_realized = rng.dirichlet(m*f_0)
    #     f_scores = h_helper(f_realized)
    #     count_correct += (np.argmax(f_scores) == max_index)
    
    # print(f"Probability of correct choice = {count_correct/prior_update_realizations*100}")
