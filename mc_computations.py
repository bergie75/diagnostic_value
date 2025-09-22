# math and user-defined
import numpy as np
from scipy.special import betainc
import matplotlib.pyplot as plt
from example_with_maximum import exported_parameters
# logistical
import os
import itertools
from multiprocessing import Pool

# the generator used for our experiments, can set a seed for consistency
rng = np.random.default_rng()

def res_ben_matrix(parameters=exported_parameters):
    num_treatments = parameters["num_treatments"]
    num_pathogens = parameters["num_pathogens"]
    resistance = parameters["resistance"]
    cost_per_qaly = parameters["cost_per_qaly"]
    qaly_if_res = parameters["qaly_if_res"]
    qaly_if_susc = parameters["qaly_if_susc"]
    cost_res = parameters["cost_res"]

    res_benefit_matrix = np.zeros((num_treatments, num_pathogens))
    for i in range(0, num_treatments):
      for j in range(0, num_pathogens):
            r_ij = resistance[i,j]
            res_benefit_matrix[i,j] = r_ij*(cost_per_qaly*qaly_if_res[i]-cost_res[j])+(1-r_ij)*cost_per_qaly*qaly_if_susc[i]
    
    return res_benefit_matrix
            
def h_helper(f, parameters=exported_parameters):
    drug_costs = parameters["drug_costs"]
    res_benefit_matrix = res_ben_matrix(parameters=parameters)
    return -drug_costs + np.matmul(res_benefit_matrix, f)  # holds values of treatments, begin by subtracting costs

def opt_net_benefits(parameters=exported_parameters):
    optimal_net_benefits_list = []
    num_pathogens = parameters["num_pathogens"]

    for j in range(0, num_pathogens):
        e_j = np.eye(1,num_pathogens,j).reshape(-1,1)  # create a standard basis vector representing all disease being caused by jth pathogen
        benefits = h_helper(e_j, parameters=parameters)
        optimal_net_benefits_list.append(np.max(benefits))

    return np.array(optimal_net_benefits_list)

# returns the argument of the best empirical diagnosis given that the believed distribution is f_realized
def argmax_helper(f_realized):
    return np.argmax(h_helper(f_realized))

def objective_function_for_2_by_2(p_test, parameters=exported_parameters, print_update=False):
    # unpack needed values from parameter list
    drug_costs = parameters["drug_costs"]
    cost_test = parameters["cost_test"]
    diagnostic_realizations = parameters["diagnostic_realizations"]
    num_patients = parameters["num_patients"]
    num_treatments = parameters["num_treatments"]
    prior_alpha = parameters["m"]*parameters["f_0"]
    p_ignore = parameters["p_ignore"]
    f_true = parameters["f_true"]
    
    # compute useful intermediate quantities with parameter list
    res_benefit_matrix = res_ben_matrix(parameters)
    optimal_net_benefits = opt_net_benefits(parameters)
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

def objective_function_for_2_by_2_parallel(p_test, parameters=exported_parameters, print_update=False):
    # unpack needed values from parameter list
    drug_costs = parameters["drug_costs"]
    cost_test = parameters["cost_test"]
    diagnostic_realizations = parameters["diagnostic_realizations"]
    num_patients = parameters["num_patients"]
    num_treatments = parameters["num_treatments"]
    prior_alpha = parameters["m"]*parameters["f_0"]
    p_ignore = parameters["p_ignore"]
    f_true = parameters["f_true"]
    
    # compute useful intermediate quantities with parameter list
    res_benefit_matrix = res_ben_matrix(parameters)
    optimal_net_benefits = opt_net_benefits(parameters)
    true_scores = h_helper(f_true)
    
    # an accumulator for the empirical treatment part of the objective function
    emp_net_benefit_estimate = 0

    # compute thresholds for comparisons
    thresh_num = drug_costs[0]-drug_costs[1]+res_benefit_matrix[1,1]-res_benefit_matrix[0,1]
    thresh_denom = res_benefit_matrix[0,0]+res_benefit_matrix[1,1]-res_benefit_matrix[0,1]-res_benefit_matrix[1,0]
    threshold = thresh_num/thresh_denom

    def simulate_diagnostics():
        # each thread gets own rng
        local_rng = np.random.default_rng()
        # one run of testing p_test of our patients to improve empirical diagnoses
        other_diagnostics = local_rng.multinomial((num_patients-1)*p_test, f_true)
        prob_estimates = np.zeros(num_treatments)  # will hold probability of treatment being best
        # update prior based on the results of our randomly generated measurements
        posterior_alpha = prior_alpha + other_diagnostics

        # use theoretical results to skip inner loop
        prob_estimates[1] = betainc(posterior_alpha[0],posterior_alpha[1],threshold)
        prob_estimates[0] = 1-prob_estimates[1]
        
        return np.dot(prob_estimates, true_scores)
    
    # perform Monte-Carlo sampling
    with Pool(diagnostic_realizations) as p:
        parallel_diagnostics = p.map(simulate_diagnostics, range(0, diagnostic_realizations))
        
    emp_net_benefit_estimate = np.mean(parallel_diagnostics)
    testing_net_benefit = np.dot(f_true, optimal_net_benefits)-cost_test
    obj_value = (1-(1-p_ignore)*p_test)*emp_net_benefit_estimate + (1-p_ignore)*p_test*testing_net_benefit - p_test*p_ignore*cost_test

    if print_update:
        print(f"Completed calculation for {p_test}")

    return obj_value

# the objective function for the restricted scenario when a physician only considers two possible
# pathogen distributions. dis1_prevalences holds the amount of pathogen 1 for each distribution
def analytic_objective(p_test, dis1_prevalences, weights=np.array([0.5, 0.5]), parameters=exported_parameters):
    # unpack needed values from parameter list
    cost_test = parameters["cost_test"]
    num_patients = parameters["num_patients"]

    # compute useful intermediate quantities with parameter list
    optimal_net_benefits = opt_net_benefits(parameters)

    # expand input parameters to full distributions in physician's prior
    f_a = np.array([dis1_prevalences[0], 1-dis1_prevalences[0]])
    f_b = np.array([dis1_prevalences[1], 1-dis1_prevalences[1]])
    f_true = parameters["f_true"]

    # compute all necessary empirical helper function values
    h_true = h_helper(f_true, parameters=parameters)
    h_a = h_helper(f_a, parameters=parameters)
    h_b = h_helper(f_b, parameters=parameters)

    # compute a quantity that determines how the 50-50 prior is updated
    # the logic of this function is not currently designed for cases
    # when the true prior is entirely concentrated on one pathogen, and f_a is as well
    if 0 < f_a[0] < 1:
        r = (f_b[0]/f_a[0])**(f_true[0])*(f_b[1]/f_a[1])**(f_true[1])
    else:
        r=np.inf

    # updated posterior probability that the physician believes the true distribution is a
    # avoid overflow errors from large numbers
    probability_actually_a = weights[0]*r**(-num_patients*p_test)/(weights[1]+weights[0]*r**(-num_patients*p_test))

    # add private value
    obj_val = p_test*np.dot(f_true, optimal_net_benefits-cost_test)

    # these two if statements determine the treatments the doctor believes to be optimal
    # and adds the corresponding expected value
    if h_a[0] >= h_a[1]:
        obj_val += (1-p_test)*probability_actually_a*h_true[0]
    else:
        obj_val += (1-p_test)*probability_actually_a*h_true[1]

    if h_b[0] >= h_b[1]:
        obj_val += (1-p_test)*(1-probability_actually_a)*h_true[0]
    else:
        obj_val += (1-p_test)*(1-probability_actually_a)*h_true[1]

    return obj_val

def find_critical_cost(parameters=exported_parameters):
    f_true = parameters["f_true"]
    optimal_net_benefits = opt_net_benefits(parameters=exported_parameters)
    
    return np.dot(f_true, optimal_net_benefits)

def run_experiment(experiment_name, vars_changed, changes_to_try, local_params=exported_parameters):
    # preliminaries to save the results
    cwd = os.getcwd()
    save_to = os.path.join(cwd,"diagnostic_value","longer_runs", experiment_name)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # generate testing frequencies
    num_patients = exported_parameters["num_patients"]  # a constant from our configuration file
    p_test_vals = np.array(range(0, num_patients+1))/num_patients

    # generate tests
    if len(changes_to_try) > 1:
        combinations = list(itertools.product(*changes_to_try))
    else:
        combinations = list(itertools.product(changes_to_try[0]))
    
    failed_saves = 0
    for combo in combinations:
        # prepare to save results to specific file and change local variables for experiment
        print(f"Working on combination={combo}")
        label = ""
        
        for i,var_name in enumerate(vars_changed):
            label += f"{var_name}_{combo[i]}_"
            local_params[var_name] = combo[i]
        
        label = label[:-1]  # strip trailing underscore

        # carry out experiment and add results to plot
        obj_fun_vals = np.array([objective_function_for_2_by_2(p_test, parameters=local_params) for p_test in p_test_vals])
        plt.plot(p_test_vals, obj_fun_vals)
        try:
            np.save(os.path.join(save_to, label), obj_fun_vals)  # save results to view later
        except:
            failed_saves += 1
            np.save(os.path.join(save_to, f"failed_save_{failed_saves}"), obj_fun_vals)
            print(f"Combination {combo} saved under failed_save_{failed_saves}")

    plt.title("Tradeoffs between public and private value of diagnostics")
    plt.xlabel("Net benefit ($)")
    plt.ylabel("Sum of public and private net benefit")
    plt.legend([f"{combo}" for combo in combinations])
    plt.show()

# could probably rewrite this with regex if needed
def plot_results(experiment_name, varying, constant_vars=[], constant_vals=[], varying_vals=None, given_labels=None):
    cwd = os.getcwd()
    prefix_folder = os.path.join(cwd,"diagnostic_value","longer_runs",experiment_name)
    
    # filter the directory to find relevant files
    filters = [f"{constant_vars[i]}_{constant_vals[i]}" for i in range(0,len(constant_vars))]
    # only allow files which mention the intended values to plot, use all if unspecified
    if varying_vals is not None:
        for value in varying_vals:
            filters.append(f"{varying}_{value}")
    else:
        filters.append(varying)
    
    if given_labels is None:
        legend_labels = []
    else:
        legend_labels = given_labels
    
    for result in os.listdir(prefix_folder):
        include = False
        for filter in filters:
            include = (include or (result.find(filter) != -1))
        if include:
            obj_vals = np.load(os.path.join(prefix_folder,result))
            p_vals = np.linspace(0,1,len(obj_vals))
            argmax_index = int(np.argmax(obj_vals))
            plt.plot(p_vals, obj_vals)
            plt.scatter(p_vals[argmax_index], obj_vals[argmax_index], color="r", label="_nolegend_")
            
            if given_labels is None:
                numpy_starts = result.find(".npy")  # used to generate legend label
                legend_labels.append(result[:numpy_starts])
                
    plt.xlabel("probability of diagnostic")
    plt.ylabel("expected net benefit per patient ($)")
    plt.legend(legend_labels)    
    plt.show()

if __name__ == "__main__":
    # vars_changed = ["f_true"]
    changes_to_try = [[np.array([0.1,0.9]), np.array([0.3,0.7]), np.array([0.5,0.5]), np.array([0.7,0.3]), np.array([0.9,0.1])]]
    # run_experiment("revised_distribution_bias", vars_changed, changes_to_try)
    #choices = [0.8,1.0,1.2,1.4]
    choices=[f"[{x[0]} {x[1]}]" for x in changes_to_try[0]]
    labels = [f"{int(100*x[0])}%" for x in changes_to_try[0]]
    print(choices)
    plot_results("revised_distribution_bias","f_true",varying_vals=choices,given_labels=labels)