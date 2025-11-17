from mc_computations import *
from example_with_maximum import exported_parameters

#prev_seq = [0.9, 0.84, 0.78, 0.72, 0.66, 0.6, 0.54, 0.6, 0.66, 0.72, 0.78, 0.84, 0.9]
delta = 0.015
prev_seq = [0.9-i*delta for i in range(0, 6)]
backward = [0.9+(i-6)*delta for i in range(0, 7)]
prev_seq.extend(backward)
print(prev_seq)

experiment_name = "slower_prevalence_variation_temporal"
exported_parameters["resistance"] = np.array([[140/189, 1],[1, 39/57]])
run_temporal_experiment(experiment_name, prev_seq, local_params=exported_parameters)
    