from mc_computations import *
from example_with_maximum import exported_parameters

delta = 0.06
prev_seq = [0.9-i*delta for i in range(0, 6)]
backward = [0.9+(i-6)*delta for i in range(0, 7)]
prev_seq.extend(backward)
print(prev_seq)

experiment_name = "faster_prevalence_variation_temporal_old_resistance"
#exported_parameters["resistance"] = np.array([[140/189, 1],[1, 39/57]])
#run_temporal_experiment(experiment_name, prev_seq, local_params=exported_parameters)
run_temporal_experiment(experiment_name, prev_seq, local_params=exported_parameters, prior_decay_rate=5/1000)