import numpy as np

# ____________________________________________________________________________
# TOTALS
# ____________________________________________________________________________
num_patients = 10**3  # patients seen in the relevant time window
num_treatments = 2  # number of different drugs available
num_pathogens = 2 # number of disease-causing agents to consider

# ____________________________________________________________________________
# MCMC control parameters
# ____________________________________________________________________________
diagnostic_realizations = 10**3
prior_update_realizations = 10**3


# ____________________________________________________________________________
# BEHAVIORAL PARAMETERS
# ____________________________________________________________________________
p_ignore = 0.01 # probability that a test is ignored after being ordered

# ____________________________________________________________________________
# DISTRIBUTIONS AND PRIORS
# ____________________________________________________________________________
f_true = np.array([0.3,0.7])  # true species breakdown
m = 10
f_0 = np.array([0.2, 0.8])

# ____________________________________________________________________________
# COSTS AND BENEFITS
# ____________________________________________________________________________
drug_costs = np.array([10,20])  # cost in dollars of drugs
cost_per_qaly = 1  # conversion factor between expected outcomes and dollars
qaly_if_susc = 3  # benefit to patient in qaly, could adjust to months
qaly_if_res = 0  # set to zero for now, could increase if there is residual symptom relief followed by resurgence
cost_res = 5  # cost of follow-on treatment due to resistance
cost_test = 2 # cost of giving patient a diagnostic

# ____________________________________________________________________________
# PATHOGEN/DRUG DYNAMICS
# ____________________________________________________________________________

# each row represents a drug, each column a bug. Entry is resistance prob. of bug to drug
# note that these do not need to sum to one along rows or columns
resistance = np.array([[0.1, 0.6],[0.55, 0.05]])

# ____________________________________________________________________________
# OPTIMAL NET BENEFIT GIVEN TESTING
# ____________________________________________________________________________
optimal_net_benefits = []

# loops to populate optimal benefits
for j in range(0, num_pathogens):
    # variable to hold current best choice
    max_benefit = -np.inf
    
    for i in range(0, num_treatments):
            r_ij = resistance[i,j]
            benefit_under_i = r_ij*(cost_per_qaly*qaly_if_res-cost_res)+(1-r_ij)*(cost_per_qaly*qaly_if_susc)-drug_costs[i]
            if benefit_under_i > max_benefit:
                  max_benefit = benefit_under_i
    
    # after defining all possible benefits, select the best drug for treating the given pathogen
    optimal_net_benefits.append(max_benefit)