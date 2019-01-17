import glob
import os
import pandas as pd
import pickle
import random
import string

SERIES_DICT = {"t1": ["MPRAGE", "T1W"], "t2": ["FLAIR", "t2_"], "ir": ["IR-EPI"]}
BASE_DIR = os.getcwd()
LOCATION_DICT = {
    "raw": os.path.join(BASE_DIR, "Scans"),
    "skull_stripped": os.path.join(BASE_DIR, "Skull-stripped"),
    "bias_corrected": os.path.join(BASE_DIR, "Bias-corrected"),
    "target": os.path.join(BASE_DIR, "target"),
}
REALIGNED_DIR_NAME = "Realigned"
TARGET_FILE_NAME = "MPRAGE.nii.gz"
COSTS_FILE_NAME = "realignment_costs.pkl"
MUTUAL_INFORMATION_FILE_NAME = "mutual_information.pkl"


def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def anonymize_data():
    subject_dirs = glob.glob(os.path.join(LOCATION_DICT["raw"], "*"))
    associations = dict()
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        new_id = id_generator()
        dest = subject_dir.replace(subject_id, new_id)
        os.rename(subject_dir, dest)
        associations[subject_id] = new_id
    target_dirs = glob.glob(os.path.join(LOCATION_DICT["target"], "*"))
    for target_dir in target_dirs:
        target_id = os.path.basename(target_dir)
        new_id = associations.get(target_id)
        if new_id:
            dest = target_dir.replace(target_id, new_id)
            os.rename(target_dir, dest)
        else:
            print(f"WARNING! Could not determine {target_id} source subject!")
    with open("associations.pkl", "wb") as key_file:
        pickle.dump(associations, key_file)
    return associations


def get_target_subject_dir(target_id: str):
    target_dir = LOCATION_DICT["target"]
    return os.path.join(target_dir, target_id)


def get_target_scan(target_id: str):
    target_subject_dir = get_target_subject_dir(target_id)
    target_scan_path = os.path.join(target_subject_dir, TARGET_FILE_NAME)
    if os.path.isfile(target_scan_path):
        return os.path.join(target_subject_dir, TARGET_FILE_NAME)
    else:
        raise FileNotFoundError(f"Failed to locate target scan in {target_scan_path}")


def get_realigned_scans_dir(target_id: str):
    target_subject_dir = get_target_subject_dir(target_id)
    return os.path.join(target_subject_dir, REALIGNED_DIR_NAME)


def format_cost_function_name(cost_function: str):
    return cost_function.replace(" ", "")


def get_cost_function_dir(target_id: str, cost_function: str):
    realigned_scans_dir = get_realigned_scans_dir(target_id)
    cost_function = format_cost_function_name(cost_function)
    return os.path.join(realigned_scans_dir, cost_function)


def get_costs_file_path(target_id: str, cost_function: str):
    cost_function_dir = get_cost_function_dir(target_id, cost_function)
    return os.path.join(cost_function_dir, COSTS_FILE_NAME)


def get_mutual_information_file_path(target_id: str, cost_function: str):
    cost_function_dir = get_cost_function_dir(target_id, cost_function)
    return os.path.join(cost_function_dir, MUTUAL_INFORMATION_FILE_NAME)


def get_realigned_subject_dir(target_id: str, cost_function: str, subject_id: str):
    cost_function_dir = get_cost_function_dir(target_id, cost_function)
    return os.path.join(cost_function_dir, subject_id)


def get_realigned_subject_data(
    target_id: str, cost_function: str, subject_id: str, include_mat: bool = True
):
    realigned_data_dir = get_realigned_subject_dir(target_id, cost_function, subject_id)
    files = glob.glob(os.path.join(realigned_data_dir, "*"))
    realigned_scan = [f for f in files if f.endswith(".nii.gz")]
    if realigned_scan:
        if include_mat:
            mat_file = [f for f in files if f.endswith(".mat")]
            if mat_file:
                return realigned_scan[0], mat_file[0]
            else:
                return realigned_scan[0], None
        else:
            return realigned_scan[0]
    return None


def generate_subject_dirs(
    status: str, target_id: str = None, cost_function: str = None
):
    if status in ["raw", "skull_stripped"]:
        pattern = os.path.join(LOCATION_DICT[status], "*/")
        return glob.iglob(pattern)
    elif status in ["realigned"]:
        realigned_scans_dir = get_cost_function_dir(target_id, cost_function)
        pattern = os.path.join(realigned_scans_dir, "*/")
        return glob.iglob(pattern)


def generate_scans(status: str, target_id: str = None, cost_function: str = None):
    if status in ["raw", "skull_stripped"]:
        pattern = os.path.join(LOCATION_DICT[status], "**/*.nii.gz")
        return glob.iglob(pattern)
    elif status in ["realigned"]:
        realigned_scans_dir = get_cost_function_dir(target_id, cost_function)
        pattern = os.path.join(realigned_scans_dir, "**/*.nii.gz")
        return glob.iglob(pattern)


def filter_scans_by_type(scans: list, scan_type: str):
    return [
        scan
        for scan in scans
        if any(
            identifier in os.path.basename(scan)
            for identifier in SERIES_DICT[scan_type]
        )
    ]


def choose_single_anatomical(scans: list):
    choice = [
        scan
        for scan in scans
        if scan.endswith("EnhancedContrast.nii.gz")
        or scan.endswith("EnchancedContrast.nii.gz")
    ]
    if not choice:
        choice = [scan for scan in scans if scan.endswith("1mm.nii.gz")]
        if not choice:
            return scans[0]
        else:
            return choice[0]
    else:
        return choice[0]


def choose_single_scan(scans: list, scan_type: str):
    if scan_type == "t1":
        return choose_single_anatomical(scans)
    else:
        return scans[0]


def get_scans(base_dir: str, scan_type: str = None, single: bool = True) -> list:
    print(f"Looking for {scan_type} scans in {base_dir}...")
    result = []
    subject_dirs = glob.glob(os.path.join(base_dir, "*/"))
    for subject_dir in subject_dirs:
        subject_id = subject_dir.split("/")[-2]
        print(f"Checking {subject_id}...")
        scans = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
        if scan_type is not None:
            scans = filter_scans_by_type(scans, scan_type)
        if scans:
            print(f"Found {len(scans)} {scan_type} scans for {subject_id}.")
            if single:
                choice = choose_single_scan(scans, scan_type)
                result += [choice]
                print(
                    f"{os.path.basename(choice)} added to {scan_type or 'scan'} list."
                )
            else:
                result += scans
                print(f"{len(scans)} scans added to list.")
        else:
            print(f"Could not find {scan_type} scan for subject {subject_id}!")
    print(f"Found {len(result)} {scan_type} scans from {len(subject_dirs)} subjects.")
    return result


COST_FUNCTION_DICT = {
    "CorrelationRatio": "Correlation Ratio",
    "LeastSquares": "Least Squares",
    "MutualInformation": "Mutual Information",
    "NormalizedCorrelation": "Normalized Correlation",
    "NormalizedMutualInformation": "Normalized Mutual Information",
}


def get_results(target_id: str, cost_function: str):
    path = get_mutual_information_file_path(target_id, cost_function)
    with open(path, "rb") as f:
        return pickle.load(f)


def get_all_results(target_id: str):
    results = [
        get_results(target_id, cost_function)
        for cost_function in COST_FUNCTION_DICT.values()
    ]
    return pd.concat(results, axis=1)
