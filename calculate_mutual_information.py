import glob
import nibabel as nib
import numpy as np
import os
import pandas as pd
import pickle
import sklearn.metrics

from dao import LOCATION_DICT, get_scans


REALIGNED_DIR_NAME = "Realigned"
FILE_NAME = "MPRAGE.nii.gz"


def get_subject_dir(subject_id: str, target_dir: str = LOCATION_DICT["target"]):
    return os.path.join(target_dir, subject_id)


def get_target_scan(
    subject_id: str,
    target_dir: str = LOCATION_DICT["target"],
    file_name: str = FILE_NAME,
):
    return os.path.join(get_subject_dir(subject_id), file_name)


def get_realigned_dir(
    subject_id: str,
    target_dir: str = LOCATION_DICT["target"],
    realigned_dir: str = REALIGNED_DIR_NAME,
):
    return os.path.join(get_subject_dir(subject_id), realigned_dir)


def default_destination(
    subject_id: str,
    cost_function: str,
    target_dir: str = LOCATION_DICT["target"],
    realigned_dir_name: str = REALIGNED_DIR_NAME,
):
    return os.path.join(get_realigned_dir(subject_id), cost_function.replace(" ", ""))


def extract_series_data(series_path: str):
    return nib.load(series_path).get_data().flatten()


def calculate_mutual_information(
    array1: np.ndarray, array2: np.ndarray, bins: int = 10
) -> np.float64:
    histogram = np.histogram2d(array1, array2, bins)[0]
    mi = sklearn.metrics.mutual_info_score(None, None, contingency=histogram)
    return mi


def calculate_series_mutual_information_score(
    series1_path: str, series2_path: str
) -> np.float64:
    series1_data = extract_series_data(series1_path)
    series2_data = extract_series_data(series2_path)
    return calculate_mutual_information(series1_data, series2_data)


def calculate_mutual_information_scores(
    base_dir: str, cost_function: str, subject_id: str
) -> pd.Series:
    target_scan = get_target_scan(subject_id)
    paths = get_scans(default_destination(subject_id, cost_function), "t1")
    mutual_information = dict()
    for path in paths:
        subject_id = path.split("/")[-2]
        mutual_information[subject_id] = calculate_series_mutual_information_score(
            path, target_scan
        )
    result = pd.Series(mutual_information)
    result.index.name = "SubjectID"
    result.name = f"{cost_function}_MutualInformation"
    return result

