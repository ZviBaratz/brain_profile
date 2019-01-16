import os

from dao import get_scans, LOCATION_DICT, get_target_scan, get_cost_function_dir
from nipype.interfaces.fsl import FLIRT, FNIRT


SERIES_TYPE = "t1"
COST_FUNCTION_DICT = {
    "Mutual Information": "mutualinfo",
    "Correlation Ratio": "corratio",
    "Normalized Correlation": "normcorr",
    "Normalized Mutual Information": "normmi",
    "Least Squares": "leastsq",
    "Boundary-Based Registration": "bbr",
}


def create_results_directory(target_id: str, cost_function: str):
    results_location = get_cost_function_dir(target_id, cost_function)
    print(f"Creating output directory in {results_location}...", end="\t")
    if os.path.isdir(results_location):
        print(f"\nOutput directory already exists in {results_location}!")
    else:
        os.makedirs(results_location, exist_ok=False)
    return results_location


def run_realign(target_id: str, cost_function: str):
    target_scan = get_target_scan(target_id)
    scans = get_scans(LOCATION_DICT["skull_stripped"], "t1")
    output_dir = create_results_directory(target_id, cost_function)
    for scan in scans:
        subject_id = scan.split("/")[-2]
        subject_results_dir = os.path.join(output_dir, subject_id)
        try:
            os.makedirs(subject_results_dir, exist_ok=False)
        except FileExistsError:
            print(f"Results for {subject_id} found! Skipping...")
            continue
        scan_name = os.path.basename(scan).split(".")[0]
        print(f"Registering {subject_id}'s {scan_name} to target...", end="\t")
        flirt = FLIRT()
        flirt.inputs.in_file = scan
        flirt.inputs.reference = target_scan
        flirt.inputs.cost = COST_FUNCTION_DICT[cost_function]
        flirt.inputs.out_file = os.path.join(subject_results_dir, f"{scan_name}.nii.gz")
        flirt.inputs.out_matrix_file = os.path.join(
            subject_results_dir, f"{scan_name}.mat"
        )
        flirt.run()
        print("done!")


def run_nonlinear_registration(target_id: str):
    target_scan = get_target_scan(target_id)
    scans = get_scans(LOCATION_DICT["skull_stripped"], "t1")
    output_dir = create_results_directory(target_id, "NonlinearSSD")

    for scan in scans:
        subject_id = scan.split("/")[-2]
        subject_results_dir = os.path.join(output_dir, subject_id)
        try:
            os.makedirs(subject_results_dir, exist_ok=False)
        except FileExistsError:
            if os.listdir(subject_results_dir):
                print(f"Results for {subject_id} found! Skipping...")
                continue
        scan_name = os.path.basename(scan).split(".")[0]
        print(f"Registering {subject_id}'s {scan_name} to target...", end="\t")
        fnirt = FNIRT()
        fnirt.inputs.in_file = scan
        fnirt.inputs.ref_file = target_scan
        fnirt.inputs.warped_file = os.path.join(
            subject_results_dir, f"{scan_name}_warped.nii.gz"
        )
        fnirt.inputs.field_file = os.path.join(
            subject_results_dir, f"{scan_name}_warped.nii.gz"
        )
        fnirt.run()
        print("done!")
