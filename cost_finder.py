import os
import pickle
import shutil

from dao import (
    get_costs_file_path,
    get_realigned_subject_data,
    generate_subject_dirs,
    get_target_scan,
)
from nipype.interfaces.fsl import FLIRT


def serialize_results(target_id: str, cost_function: str, results: dict) -> bool:
    file_path = get_costs_file_path(target_id, cost_function)
    with open(file_path, "wb") as results_file:
        pickle.dump(results, results_file)
        return True


def calculate_realignment_cost(
    target_id: str, cost_function: str, serialize: bool = True
):
    realigned_subject_dirs = generate_subject_dirs(
        "realigned", target_id, cost_function
    )
    target_scan = get_target_scan(target_id)
    costs = {}
    for subject_dir in realigned_subject_dirs:
        subject_id = subject_dir.split("/")[-2]
        registered, mat_file = get_realigned_subject_data(subject_id)
        print(f"Calculating cost function value...", end="\t")
        flirt = FLIRT()
        flirt.inputs.in_file = registered
        flirt.inputs.reference = target_scan
        flirt.inputs.schedule = "/usr/local/fsl/etc/flirtsch/measurecost1.sch"
        flirt.inputs.in_matrix_file = mat_file
        tmp = os.path.join(subject_dir, "tmp")
        flirt.inputs.out_file = os.path.join(tmp, "cost.nii.gz")
        flirt.inputs.out_matrix_file = os.path.join(tmp, "cost.mat")
        os.makedirs(tmp, exist_ok=True)
        f = flirt.run()
        result = float(f.runtime.stdout.split()[0])
        print(f"done! [{result}]")
        shutil.rmtree(tmp)
        costs[subject_id] = result
        if serialize:
            serialize_results(target_id, cost_function, costs)
    return costs
