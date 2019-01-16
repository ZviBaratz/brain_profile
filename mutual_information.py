import nibabel as nib
import numpy as np
import pandas as pd
import pickle
import sklearn.metrics

from dao import (
    COST_FUNCTION_DICT,
    get_target_scan,
    generate_subject_dirs,
    get_realigned_subject_data,
    get_mutual_information_file_path,
    get_all_results,
)


def extract_series_data(series_path: str):
    return nib.load(series_path).get_data().flatten()


def calculate_mutual_information(
    array1: np.ndarray, array2: np.ndarray, bins: int = 10
) -> np.float64:
    histogram = np.histogram2d(array1, array2, bins)[0]
    mi = sklearn.metrics.mutual_info_score(None, None, contingency=histogram)
    return mi


def calculate_scans_mutual_information_score(
    scan1_path: str, scan2_path: str
) -> np.float64:
    scan1_data = extract_series_data(scan1_path)
    scan2_data = extract_series_data(scan2_path)
    return calculate_mutual_information(scan1_data, scan2_data)


def mutual_information_dict_to_series(
    mutual_information: dict, cost_function: str
) -> pd.Series:
    result = pd.Series(mutual_information)
    result.index.name = "Subject ID"
    result.name = f"Mutual Information Score After {cost_function} Realignment"
    return result


def calculate_mutual_information_scores(
    target_id: str, cost_function: str, as_series: bool = True, serialize: bool = True
) -> pd.Series:
    target_scan = get_target_scan(target_id)
    realigned_subject_dirs = generate_subject_dirs(
        "realigned", target_id, cost_function
    )
    mutual_information = dict()
    print(
        f"\n\u0FD4 Calculating mutual information scores for target {target_id} after {cost_function} realignment \u0FD4\n"
    )
    for subject_dir in realigned_subject_dirs:
        subject_id = subject_dir.split("/")[-2]
        print(f"Calculating mutual information for {subject_id}...", end="\t")
        realigned_scan_path = get_realigned_subject_data(
            target_id, cost_function, subject_id, include_mat=False
        )
        mutual_information_score = calculate_scans_mutual_information_score(
            target_scan, realigned_scan_path
        )
        mutual_information[subject_id] = mutual_information_score
        print(f"\u2714\t[{mutual_information_score}]")

    if as_series:
        print("Converting to pandas series object...", end="\t")
        series = mutual_information_dict_to_series(mutual_information, cost_function)
        print("\u2714")
        if serialize:
            file_path = get_mutual_information_file_path(target_id, cost_function)
            print(f"Saving to {file_path}...", end="\t")
            series.to_pickle(file_path)
            print("\u2714")
        return series
    else:
        if serialize:
            file_path = get_mutual_information_file_path(target_id, cost_function)
            print(f"Saving to {file_path}...", end="\t")
            with open(file_path, "wb") as mutual_information_file:
                pickle.dump(mutual_information, mutual_information_file)
            print("\u2714")
        return mutual_information


def calculate_all_mutual_information_scores(target_id: str) -> pd.DataFrame:
    for cost_function in COST_FUNCTION_DICT.values():
        calculate_mutual_information_scores(target_id, cost_function)
    return get_all_results(target_id)
