# math and user-defined
import numpy as np
from math import floor, ceil
from scipy.special import betainc, beta
import matplotlib.pyplot as plt
from example_with_maximum import exported_parameters
# logistical
import os
import copy
import itertools
from multiprocessing import Pool
import re
import string

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

def dirichlet_pdf(x, alpha, tol=10**(-10)):
    # support defined only for values in the probability simplex
    if abs(1-np.sum(x)) > tol:
        return 0
    
    pdf = 1/beta(*alpha)
    for i in range(0, len(x)):
        pdf *= x[i]**(alpha[i]-1)
    return pdf

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
    # this handles the edge case where both treatments are precisely equally effective and costly
    # for all intents and purposes, only the "if" case matters
    if thresh_denom!=0.0 or thresh_num!=0.0:
        threshold = thresh_num/thresh_denom
    else:
        threshold = 0

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

def obj_fun_with_wastewater(p_test, parameters=exported_parameters,
                  use_wes=True, amb_conf=0.1, amb_mean=[0.5, 0.5]):
    obj_fun = 0
    for _ in range(0, parameters["prior_update_realizations"]):
        local_params = copy.deepcopy(parameters)
        # alter the planner's prior according to wes output (or lack thereof)
        if use_wes:
            wes_sample = rng.dirichlet(local_params["wes_prior"])
            local_params["m"] = np.sum(wes_sample)
            local_params["f_0"] = wes_sample/local_params["m"]
        else:
            local_params["m"] = amb_conf
            local_params["f_0"] = amb_mean
        
        obj_fun = objective_function_for_2_by_2(p_test, parameters=local_params)
    obj_fun /= parameters["prior_update_realizations"]
    
    return obj_fun

def golden_search(min_patients, max_patients, parameters=exported_parameters):
    invphi=0.618  # golden ratio, used for optimal selection of intervals
    num_patients = exported_parameters["num_patients"]
    
    to_maximize = lambda p: obj_fun_with_wastewater(p, use_wes=True, parameters=parameters)
    old_min = -np.inf
    old_max = np.inf

    while (max_patients - min_patients > 1) and ((min_patients-old_min>0) or (old_max-max_patients>0)):
        old_min = min_patients
        old_max = max_patients
        print(f"Current bracket: [{min_patients, max_patients}]")
        interior1 = floor(max_patients - (max_patients-min_patients)*invphi)
        interior2 = ceil(min_patients + (max_patients-min_patients)*invphi)
        f_at_1 = to_maximize(interior1)
        f_at_2 = to_maximize(interior2)
        if f_at_1 > f_at_2:
            max_patients = interior2
        else:
            min_patients = interior1
    
    return (min_patients + max_patients)/(2*num_patients)

def secant_search(starting_patients, min_patients, max_patients, parameters=exported_parameters,
                  stepsize=0.01, error_tol=10**(-6), step_decay=0.99):
    
    num_patients = exported_parameters["num_patients"]
    to_maximize = lambda p: obj_fun_with_wastewater(p, use_wes=True, parameters=parameters)
    
    current_patients = starting_patients
    current_val = to_maximize(starting_patients/num_patients)
    
    old_patients = np.inf
    old_val = -np.inf

    while (np.abs(current_patients-old_patients) > 1) and np.abs(current_val-old_val) > error_tol:
        print(f"Current optimal estimate: {current_patients}")
        old_patients = current_patients
        old_val = current_val

        if current_patients == num_patients:
            neighboring_patients = current_patients - 1
        else:
            neighboring_patients = current_patients + 1
        
        neighboring_func_value = to_maximize(neighboring_patients/num_patients)
        secant = num_patients*(neighboring_func_value - current_val)/(neighboring_patients - current_patients)

        current_patients += round(stepsize*secant), max_patients
        current_patients = min(max(min_patients, current_patients), max_patients)
        stepsize *= step_decay
        current_val = to_maximize(current_patients/num_patients)
    
    return current_patients/num_patients

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

def private_term_only(p_test, parameters=exported_parameters, mod_vars={}):
    # allow for modifications, key for reloading past experiments I ran without this breakdown in mind
    for var_name in mod_vars.keys():
        parameters[var_name] = mod_vars[var_name]
    
    # unpack needed values from parameter list
    cost_test = parameters["cost_test"]
    f_true = parameters["f_true"]
    
    # compute useful intermediate quantities with parameter list
    optimal_net_benefits = opt_net_benefits(parameters)
    testing_net_benefit = np.dot(f_true, optimal_net_benefits)-cost_test

    return p_test*testing_net_benefit

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

def run_experiment(experiment_name, vars_changed, changes_to_try, local_params=exported_parameters,
                    plot_results=False, name_override=None):
    # preliminaries to save the results
    cwd = os.getcwd()
    save_to = os.path.join(cwd,"diagnostic_value","longer_runs", experiment_name)
    optimal_testing_frequencies = []

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # generate testing frequencies
    num_patients = exported_parameters["num_patients"]  # a constant from our configuration file
    p_test_vals = np.array(range(0, num_patients+1))/num_patients

    # generate tests, handling issues when only one variable changes
    # this depends on whether that one variable is vector-valued
    if len(changes_to_try) > 1:
        combinations = list(itertools.product(*changes_to_try))
    elif len(changes_to_try[0] > 1):
        combinations = list(itertools.product(changes_to_try))
    else:
        combinations = list(itertools.product(changes_to_try[0]))

    failed_saves = 0
    for k,combo in enumerate(combinations):
        # prepare to save results to specific file and change local variables for experiment
        print(f"Working on combination={combo}")
        label = ""
        
        for i,var_name in enumerate(vars_changed):
            label += f"{var_name}_{combo[i]}_"
            local_params[var_name] = combo[i]
        
        label = label[:-1]  # strip trailing underscore

        # carry out experiment and add results to plot
        obj_fun_vals = np.array([objective_function_for_2_by_2(p_test, parameters=local_params) for p_test in p_test_vals]) 
        argmax_index = int(np.argmax(obj_fun_vals))
        optimal_test_frequency = p_test_vals[argmax_index]
        optimal_testing_frequencies.append(optimal_test_frequency)
        
        if plot_results:
            plt.plot(p_test_vals, obj_fun_vals)
        
        # logic to save results, including a manual override of the typical naming scheme
        if name_override is not None:
            np.save(os.path.join(save_to, name_override[k]), obj_fun_vals)  # save results to view later
        else:
            try:
                np.save(os.path.join(save_to, label), obj_fun_vals)  # save results to view later
            except:
                failed_saves += 1
                np.save(os.path.join(save_to, f"failed_save_{failed_saves}"), obj_fun_vals)
                print(f"Combination {combo} saved under failed_save_{failed_saves}")

    if plot_results:
        plt.title("Tradeoffs between public and private value of diagnostics")
        plt.xlabel("Net benefit ($)")
        plt.ylabel("Sum of public and private net benefit")
        plt.legend([f"{combo}" for combo in combinations])
        plt.show()
    
    return optimal_testing_frequencies

# could probably rewrite this with regex if needed
def plot_results(experiment_name, varying, constant_vars=[], constant_vals=[], varying_vals=None,
                  given_labels=None, p_range=[0, 1], legend_title="Legend"):
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
            min_include = (p_vals >= p_range[0])
            max_include = (p_vals <= p_range[1])
            include_bools = min_include*max_include
            argmax_index = int(np.argmax(obj_vals))
            plt.plot(p_vals[include_bools], obj_vals[include_bools])
            
            if p_range[0] <= p_vals[argmax_index] <= p_range[1]:
                plt.scatter(p_vals[argmax_index], obj_vals[argmax_index], color="r", label="_nolegend_")
            
            if given_labels is None:
                numpy_starts = result.find(".npy")  # used to generate legend label
                legend_labels.append(result[:numpy_starts])
                
    # Add maximum marker to legend
    plt.scatter([],[],marker='o',color='r')
    legend_labels.append("optimum")
    
    plt.xlabel("fraction of patients given diagnostics")
    plt.ylabel("expected net benefit per patient ($)")
    plt.legend(legend_labels, title=legend_title)    
    plt.show()

def run_wes_prior_experiment(experiment_name, 
                             ambivalence_confidence_level=0.1, ambivalence_mean=np.array([0.5,0.5])):
    pass

def run_temporal_experiment(experiment_name, prev_sequence, prior_decay_rate=1,
                             local_params=exported_parameters, set_m=None, set_f_0=None):
    # will hold final testing frequencies
    optimal_testing_frequencies = []
    
    # create local fork of the exported parameters, and allow user to change starting prior
    # which is useful for resets
    period_params = local_params.copy()
    if (set_m is not None) and (set_f_0 is not None):
        period_params["m"] = set_m
        period_params["f_0"] = set_f_0
    
    # this is only needed to save the results of the priors
    save_to = os.path.join(os.getcwd(),"diagnostic_value","longer_runs")

    # the above is automatically included in the folder name for the experiment, so only this
    # needs to be added
    subpath = os.path.join(experiment_name, f"decay_rate_{prior_decay_rate:.4f}")
    
    for i,prevalence in enumerate(prev_sequence):
        f_true = np.array([prevalence,1-prevalence])
        period_params["m"] = period_params["m"]*prior_decay_rate  # to allow physician's to ignore older information
        optimal_freq = run_experiment(subpath, ["f_true"], [f_true], local_params=period_params.copy(),
                                       plot_results=False, name_override=f"time_index_{i}_prev_{prevalence}")[0]

        # after conducting the experiment for the current period, update the prior for the next period
        alpha = period_params["m"]*period_params["f_0"]+period_params["num_patients"]*optimal_freq*f_true  # update prior
        alpha_magnitude = np.sum(alpha)
        period_params["m"] = alpha_magnitude
        period_params["f_0"] = alpha/alpha_magnitude

        np.save(os.path.join(save_to, subpath, f"prior_at_index_{i}"), alpha) # used for restarts

        # add results to output
        optimal_testing_frequencies.append(optimal_freq)

    np.save(os.path.join(save_to, subpath, "optimal_sampling_rates"), np.array(optimal_testing_frequencies))
    np.save(os.path.join(save_to, subpath, "prev_sequence"), np.array(prev_sequence))
    return optimal_testing_frequencies, period_params["m"], period_params["f_0"]

def collect_temporal_results(experiment_name, prev_sequence=None, prior_decay_rate=1):
    cwd = os.getcwd()
    experiment_folder = os.path.join(cwd,"diagnostic_value","longer_runs", experiment_name, f"decay_rate_{prior_decay_rate:.4f}")

    optimal_testing_results = np.load(os.path.join(experiment_folder, "optimal_sampling_rates.npy"))
    prior_parameters = [None]*len(optimal_testing_results)

    # load saved prev sequence if available
    if prev_sequence is None:
        prev_sequence = np.load(os.path.join(experiment_folder, "prev_sequence.npy"))
    
    for file in os.listdir(experiment_folder):
        # find time index (in case there are ordering issues, use regex)
        index_string_prior = re.findall('prior_at_index_[0123456789]+', file)
        
        if len(index_string_prior) > 0:
            time_index=int(index_string_prior[0].split('_')[-1])
            alpha = np.load(os.path.join(experiment_folder, file))
            prior_parameters[time_index] = alpha

    return optimal_testing_results, prior_parameters, prev_sequence

def plot_temporal_results(experiment_name, prev_sequence=None, prior_decay_rate=1, plotting_granularity=10**3):
    # gather results
    optimal_testing_freqs, prior_parameters, prev_sequence = collect_temporal_results(experiment_name, prev_sequence, prior_decay_rate=prior_decay_rate)
    
    _, (ax1, ax2) = plt.subplots(2, 1)
    
    ax1.set_title("Optimal fraction of patients to test over time")
    ax1.set_ylabel("Optimal testing fraction")

    time_indices = list(range(0, len(prev_sequence)))
    ax1.scatter(time_indices, optimal_testing_freqs, c="red")
    
    ax2.set_title("Planner estimates of disease one prevalence vs. true prevalence")
    ax2.set_xlabel("Time index")
    ax2.set_ylabel("Disease one prevalence")
    mean_dist = []
    
    for i in range(0, len(prev_sequence)):
        mean_dist.append(prior_parameters[i][0]/np.sum(prior_parameters[i]))
    
    ax2.scatter(time_indices, mean_dist, c="red")
    ax2.scatter(time_indices, prev_sequence, c="black", marker="x")
    ax2.legend(["Planner estimate", "True prevalence"])

    plt.tight_layout()
    plt.show()
    
def public_private_breakdown(experiment_name, varying, constant_vars=[], constant_vals=[], varying_vals=None,
                             labels=None, title="", xlabel=""):
    cwd = os.getcwd()
    prefix_folder = os.path.join(cwd,"diagnostic_value","longer_runs",experiment_name)
    
    # filter the directory to find relevant files
    filters = [f"{constant_vars[i]}_{constant_vals[i]}" for i in range(0, len(constant_vars))]
    # only allow files which mention the intended values to plot, use all if unspecified
    if varying_vals is not None:
        for value in varying_vals:
            filters.append(f"{varying}_{value}")
    else:
        filters.append(varying)
    
    found_results = {}
    for result in os.listdir(prefix_folder):
        include = False
        for filter in filters:
            include = (include or (result.find(filter) != -1))
        if include:
            # need to break filter apart again to automatically alter variables used
            decimal_index = result.find(".")
            cutoff = result[decimal_index+1:].find(".")+decimal_index+1
            variable_value = result[decimal_index-1:cutoff]
            variable_name = result[:decimal_index-2]
            # compute public-private split
            obj_vals = np.load(os.path.join(prefix_folder,result))
            p_vals = np.linspace(0,1,len(obj_vals))
            argmax_index = int(np.argmax(obj_vals))
            private_val = private_term_only(p_vals[argmax_index], mod_vars={variable_name: float(variable_value)})
            public_val = obj_vals[argmax_index] - private_val
            found_results[f"{variable_name}: {variable_value}"] = [public_val, private_val]
    
    # use accumulated results to plot a graph
    num_results = len(found_results.keys())
    area_width = 1/(1+num_results)
    plot_width = 0.8*area_width/2
    plot_spots = [i*area_width for i in range(1,num_results+1)]

    for i,key in enumerate(found_results.keys()):
        if i==0:
            legend_val = ["Public value", "Private value"]
        else:
            legend_val = None
        
        plt.bar([plot_spots[i]-plot_width/2, plot_spots[i]+plot_width/2], found_results[key],
                 width=plot_width, color=["tab:red", "tab:blue"], label=legend_val)
    # use filters unless prettier labels are supplied
    if labels is None:
        labels = found_results.keys()
    plt.xticks(plot_spots, labels=labels)
    plt.ylabel("Net benefit per patient ($)")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.show()

def augmented_plot(experiment_name, varying, constant_vars=[], constant_vals=[], varying_vals=None,
                  given_labels=None, p_range=[0, 1], legend_title="Legend",
                  bar_title="", bar_xlabel="", line_title=""):
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
    
    found_results = {}  # an accumulator for the bar chart, has to be drawn later for auto-spacing
    # create subplots for figure
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    for result in os.listdir(prefix_folder):
        include = False
        for filter in filters:
            include = (include or (result.find(filter) != -1))
        if include:
            obj_vals = np.load(os.path.join(prefix_folder,result))
            p_vals = np.linspace(0,1,len(obj_vals))
            min_include = (p_vals >= p_range[0])
            max_include = (p_vals <= p_range[1])
            include_bools = min_include*max_include
            argmax_index = int(np.argmax(obj_vals))
            ax1.plot(p_vals[include_bools], obj_vals[include_bools])
            
            if p_range[0] <= p_vals[argmax_index] <= p_range[1]:
                ax1.scatter(p_vals[argmax_index], obj_vals[argmax_index], color="r", label="_nolegend_")
            
            if given_labels is None:
                numpy_starts = result.find(".npy")  # used to generate legend label
                legend_labels.append(result[:numpy_starts])
            
            # need to break filter apart again to automatically alter variables used
            decimal_index = result.find(".")
            cutoff = result[decimal_index+1:].find(".")+decimal_index+1
            variable_value = result[decimal_index-1:cutoff]
            variable_name = result[:decimal_index-2]
            # compute public-private split
            private_val = private_term_only(p_vals[argmax_index], mod_vars={variable_name: float(variable_value)})
            public_val = obj_vals[argmax_index] - private_val
            found_results[f"{variable_name}: {variable_value}"] = [public_val, private_val]

    # use accumulated results to plot a graph
    num_results = len(found_results.keys())
    area_width = 1/(1+num_results)
    plot_width = 0.8*area_width/2
    plot_spots = [i*area_width for i in range(1,num_results+1)]

    for i,key in enumerate(found_results.keys()):
        if i==0:
            legend_val = ["Public value", "Private value"]
        else:
            legend_val = None
        
        ax2.bar([plot_spots[i]-plot_width/2, plot_spots[i]+plot_width/2], found_results[key],
                 width=plot_width, color=["tab:red", "tab:blue"], label=legend_val)
    # use filters unless prettier labels are supplied
    if given_labels is None:
        given_labels = found_results.keys()

    ax2.set_xticks(plot_spots, labels=given_labels)
    ax2.set_xlabel(bar_xlabel)
    ax2.set_title(bar_title)
    ax2.text(-0.1, 1.05, string.ascii_uppercase[1], transform=ax2.transAxes, 
            size=20, weight='bold')
    ax2.legend()
    
    # do this second because legend labels are being changed by reference:
    # Add maximum marker to legend
    ax1.scatter([],[],marker='o',color='r')
    legend_labels.append("optimum")

    ax1.set_xlabel("fraction of patients given diagnostics")
    ax1.set_ylabel("expected net benefit per patient ($)")
    ax1.set_title(line_title)
    ax1.text(-0.1, 1.05, string.ascii_uppercase[0], transform=ax1.transAxes, 
            size=20, weight='bold')
    ax1.legend(legend_labels, title=legend_title)

    plt.show()

if __name__ == "__main__":
    #prev_seq = [0.9, 0.84, 0.78, 0.72, 0.66, 0.6, 0.54, 0.6, 0.66, 0.72, 0.78, 0.84, 0.9]
    plot_temporal_results("faster_prevalence_variation_temporal_old_resistance")
    experiment_name = "revised_cost_monospectral_treatments"
    varying = "cost_test"
    # experiment_values = [np.array([0.1, 0.9]), np.array([0.3, 0.7]), np.array([0.5, 0.5]),
    #                       np.array([0.7, 0.3]), np.array([0.9, 0.1])]
    chosen_values = [1.4, 2.0, 2.6, 3.2]
    exported_parameters["resistance"] = np.array([[140/189, 1],[1, 39/57]])
    # exported_parameters["cost_test"] = 3.6
    labels = [f"${x:0.2f}" for x in chosen_values]
    title = "Public and private value with varying diagnostic cost"
    xlabel = "cost of diagnostic test"
    #public_private_breakdown(experiment_name, varying, varying_vals=chosen_values, 
                             #labels=labels, title=title, xlabel=xlabel)

    # run_experiment(experiment_name, [varying], [chosen_values], use_mc=False)
    #plot_results(experiment_name, varying, varying_vals=chosen_values, given_labels=labels,
                   #legend_title="Disease one prevalence")
    # augmented_plot(experiment_name, varying, varying_vals=chosen_values, given_labels=labels,
    #                legend_title="Disease one prevalence", bar_xlabel=xlabel)
    
    # # interpolated values for resistance
    # t_vals = ["0.000", "0.02", "0.04", "0.06"]
    # #resistance_candidates = [t*base_res+(1-t)*same_resistance for t in t_vals]
    # t_labels = [f"{int(float(x)*100)}% difference" for x in t_vals]
    # t_labels[0] = "treatments are equivalent"
    # plot_results("equalized_effectiveness","t_val",varying_vals=t_vals,given_labels=t_labels,
    #              p_range=[0, 0.2],legend_title="Similarity of resistance profiles")
    