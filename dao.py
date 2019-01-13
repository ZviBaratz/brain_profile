import glob
import os
import pandas as pd
import pickle

SERIES_DICT = {"t1": ["MPRAGE", "T1W"], "t2": ["FLAIR", "t2_"], "ir": ["IR-EPI"]}


def get_scans(base_dir: str, series_type: str, single: bool = True) -> list:
    print(f"Looking for {series_type} scans in {base_dir}...")
    result = []
    subjects = glob.glob(os.path.join(base_dir, "*"))
    for subject in subjects:
        subject_id = subject.split("/")[-1]
        print(f"Checking {subject_id}...")
        scans = glob.iglob(os.path.join(subject, "*.nii.gz"))
        matching = [
            scan
            for scan in scans
            if any(
                identifier in os.path.basename(scan)
                for identifier in SERIES_DICT[series_type]
            )
        ]
        if matching:
            print(f"Found {len(matching)} {series_type} scans for {subject_id}.")
            if single:
                choice = [
                    scan
                    for scan in matching
                    if scan.endswith("EnhancedContrast.nii.gz")
                    or scan.endswith("EnchancedContrast.nii.gz")
                ]
                if not choice:
                    choice = [scan for scan in matching if scan.endswith("1mm.nii.gz")]
                    if not choice:
                        choice = matching[0]
                    else:
                        choice = choice[0]
                else:
                    choice = choice[0]
                result += [choice]
                print(f"{os.path.basename(choice)} added to {series_type} list.")
            else:
                result += matching
                print(f"{len(matching)} scans added to list.")
        else:
            print(f"Could not find {series_type} scan for subject {subject_id}!")
    print(f"Found {len(result)} {series_type} scans from {len(subjects)} subjects.")
    return result


# Read resulting cost files


def find_cost_files(base_dir: str, file_name: str):
    pattern = os.path.join(base_dir, "**", file_name)
    return glob.glob(pattern, recursive=True)


def read_costs(base_dir: str, file_name: str):
    files = find_cost_files(base_dir, file_name)
    d = {}
    for cost_file in files:
        description = cost_file.split("/")[-2]
        with open(cost_file, "rb") as results:
            d[description] = pickle.load(results)
    return pd.DataFrame.from_dict(d)


def derive_result_metric(file_path: str) -> str:
    file_name = os.path.basename(file_path)
    if file_name == "cost.pkl":
        return "Cost Estimate"
    elif file_name == "mutual_information":
        return "Mutual Information"
    else:
        return None


def derive_realignment_method(file_path: str) -> str:
    return file_path.split("/")[-3]


def derive_subject_id(file_path: str) -> str:
    return file_path.split("/")[-2]


def convert_cost_dict_to_series(path: str) -> pd.Series:
    with open(path, "rb") as f:
        d = pickle.load(f)
    series = pd.Series(d)
    method = path.split("/")[-2]
    series.index.name = "SubjectID"
    series.name = f"{method}_Cost"
    series.to_pickle(path.replace("cost", "cost_df"))
    return series


metric_ids = {"Cost": "Cost Estimate", "MutualInformation": "Mutual Information"}
method_ids = {
    "CorrelationRatio": "Correlation Ratio",
    "LeastSquares": "Least Squares",
    "MutualInformation": "Mutual Information",
    "NormalizedCorrelation": "Normalized Correlation",
    "NormalizedMutualInformation": "Normalized Mutual Information",
}


def get_results(base_dir: str):
    pattern = os.path.join(base_dir, "**/*.pkl")
    files = glob.glob(pattern)
    with open(files[0], "rb") as f:
        sample_index = pickle.load(f).index
    all_results = pd.DataFrame(
        columns=["Value"],
        index=pd.MultiIndex.from_product(
            [
                sample_index,
                [
                    "Mutual Information",
                    "Correlation Ratio",
                    "Normalized Correlation",
                    "Normalized Mutual Information",
                    "Least Squares",
                ],
                ["Cost Estimate", "Mutual Information"],
            ],
            names=["Subject ID", "Cost Function", "Metric"],
        ),
    )
    for result_file in files:
        with open(result_file, "rb") as file_content:
            results = pickle.load(file_content)
        method, metric = results.name.split("_")
        metric = metric_ids[metric]
        method = method_ids[method]
        for subject_id in results.index:
            all_results.at[(subject_id, method, metric), "Value"] = results[subject_id]
    return all_results

