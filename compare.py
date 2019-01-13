import glob
import nibabel as nib
import numpy as np
import os
import pandas as pd
import pickle
import shutil
import sklearn.metrics

from nipype.interfaces.fsl import FLIRT

BASE_DIR = "/export/home/zvibaratz/Projects/reid/target/Realigned"
DATA_DIR = (
    "/export/home/zvibaratz/Projects/reid/target/Realigned/NormalizedMutualInformation"
)
TARGET = "/export/home/zvibaratz/Projects/reid/target/MPRAGE.nii.gz"


def find_realigned_nii_and_mat(subject_dir: str):
    subject_id = subject_dir.split("/")[-2]
    print(f"Locating realigned data for {subject_id}...", end="\t")
    files = glob.glob(os.path.join(subject_dir, "*"))
    mat_file = [f for f in files if f.endswith(".mat")][0]
    registered = [f for f in files if f.endswith(".nii.gz")][0]
    print("done!")
    return registered, mat_file


def serialize_results(results: dict, data_dir: str = DATA_DIR):
    file_path = os.path.join(data_dir, "cost.pkl")
    with open(file_path, "wb") as results_file:
        pickle.dump(results, results_file)


def get_cost_function_values(base_dir: str, cost_function: str, target: str):
    data_dir = os.path.join(base_dir, cost_function)
    d = {}
    for subject_dir in glob.iglob(os.path.join(data_dir, "*/")):
        subject_id = subject_dir.split("/")[-2]
        registered, mat_file = find_realigned_nii_and_mat(subject_dir)
        print(f"Calculating cost function value...", end="\t")
        flirt = FLIRT()
        flirt.inputs.in_file = registered
        flirt.inputs.reference = target
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
        d[subject_id] = result
        serialize_results(d, data_dir)
    return d


def calculate_mutual_information(
    array1: np.ndarray, array2: np.ndarray, bins: int = 10
) -> np.float64:
    histogram = np.histogram2d(array1, array2, bins)[0]
    mi = sklearn.metrics.mutual_info_score(None, None, contingency=histogram)
    return mi


def extract_series_data(series_path: str):
    return nib.load(series_path).get_data().flatten()


def calculate_series_mutual_information_score(
    series1_path: str, series2_path: str
) -> np.float64:
    series1_data = extract_series_data(series1_path)
    series2_data = extract_series_data(series2_path)
    return calculate_mutual_information(series1_data, series2_data)


def get_realigned_series_paths(base_dir: str, cost_function: str):
    pattern = os.path.join(base_dir, cost_function, "**/*.nii.gz")
    return glob.glob(pattern)


def calculate_mutual_information_scores(
    base_dir: str, cost_function: str, target: str
) -> pd.Series:
    paths = get_realigned_series_paths(base_dir, cost_function)
    mutual_information = dict()
    for path in paths:
        subject_id = path.split("/")[-2]
        mutual_information[subject_id] = calculate_series_mutual_information_score(
            path, target
        )
    result = pd.Series(mutual_information)
    result.index.name = "SubjectID"
    result.name = f"{cost_function}_MutualInformation"
    return result

