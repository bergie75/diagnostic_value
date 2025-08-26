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
diagnostic_realizations = 10**2
prior_update_realizations = 10**2


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
#resistance = np.array([[140/189, 39/57],[183/230, 48/60]])
#resistance = np.array([[140/189, 48/60],[183/230, 39/57]])
resistance = np.array([[0.9,0.1],[0.1,0.9]])

# we combine this with previous variables to aid a function in the mc_computations file
res_benefit_matrix = np.zeros((num_treatments, num_pathogens))
for i in range(0, num_treatments):
      for j in range(0, num_pathogens):
            r_ij = resistance[i,j]
            res_benefit_matrix[i,j] = resistance[i,j]*(r_ij*(cost_per_qaly*qaly_if_res[i]-cost_res[j])+(1-r_ij)*cost_per_qaly*qaly_if_susc[i])
            
# ____________________________________________________________________________
# OPTIMAL NET BENEFIT GIVEN TESTING
# ____________________________________________________________________________
optimal_net_benefits_list = []

# loop to populate optimal benefits
for j in range(0, num_pathogens):
    benefits = res_benefit_matrix[:,j]
    optimal_net_benefits_list.append(np.max(benefits))

optimal_net_benefits = np.array(optimal_net_benefits_list)