from mc_computations import *
from example_with_maximum import exported_parameters
import numpy as np
import os

exported_parameters["resistance"] = np.array([[140/189, 1],[1, 39/57]])
num_patients = exported_parameters["num_patients"]

# we will save partially completed results as we go
cwd = os.getcwd()
save_folder = os.path.join(cwd, "diagnostic_value", "longer_runs", "example_with_wes")

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_file = os.path.join(save_folder, "objective_values")

p_vals = np.linspace(0, 1, num_patients+1)
obj_vals = np.zeros(num_patients+1)

for i,p in enumerate(p_vals):
    obj_vals[i] = obj_fun_with_wastewater(p, parameters=exported_parameters)
    if i % 10 == 0:
        print(f"Completed {100*i/len(p_vals):.2f}% of task, saving work")
        np.save(save_file, obj_vals)

# once more to ensure nothing is lost
np.save(save_file, obj_vals)
