import numpy as np

# ____________________________________________________________________________
# TOTALS
# ____________________________________________________________________________
num_patients = 10**3  # patients seen in the relevant time window
num_treatments = 2  # number of different drugs available
num_pathogens = 2 # number of disease-causing agents to consider

# ____________________________________________________________________________
# MONTE CARLO CONTROL
# ____________________________________________________________________________
diagnostic_realizations = 10**5
prior_update_realizations = 10**3


# ____________________________________________________________________________
# BEHAVIORAL PARAMETERS
# ____________________________________________________________________________
p_ignore = 0.0 # probability that a test is ignored after being ordered

# ____________________________________________________________________________
# DISTRIBUTIONS AND PRIORS
# ____________________________________________________________________________
f_true = np.array([0.9,0.1])  # true species breakdown
m = 10
f_0 = np.array([0.5, 0.5])

# ____________________________________________________________________________
# COSTS AND BENEFITS
# ____________________________________________________________________________
drug_costs = np.array([0.78,0.31])  # cost in dollars of drugs
cost_per_qaly = 53.91  # conversion factor between expected outcomes and dollars
qaly_if_susc = np.array([0.806,0.807]) # benefit to patient in qaly, could adjust to months
qaly_if_res = np.array([0.68,0.68])  # set to zero for now, could increase if there is residual symptom relief followed by resurgence
cost_res = np.array([28.98,33.79])*1.3  # cost of follow-on treatment due to resistance
cost_test = 2 # cost of giving patient a diagnostic

# ____________________________________________________________________________
# PATHOGEN/DRUG DYNAMICS
# ____________________________________________________________________________

# each row represents a drug, each column a bug. Entry is resistance prob. of bug to drug
# note that these do not need to sum to one along rows or columns
resistance = np.array([[140/189, 48/60],[183/230, 39/57]])

# ____________________________________________________________________________
# COMBINE INTO LIST
# ____________________________________________________________________________

exported_parameters_list = [num_patients, num_treatments, num_pathogens,
                       diagnostic_realizations, prior_update_realizations,
                       f_true, m, f_0,
                       drug_costs, cost_per_qaly, qaly_if_susc, qaly_if_res, cost_res, cost_test,
                       resistance, p_ignore]

par_names = ["num_patients", "num_treatments", "num_pathogens",
              "diagnostic_realizations", "prior_update_realizations",
                "f_true", "m", "f_0", "drug_costs", "cost_per_qaly", "qaly_if_susc", "qaly_if_res", "cost_res", "cost_test",
                  "resistance", "p_ignore"]

exported_parameters = dict(zip(par_names, exported_parameters_list))