import glob
import os

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
                choice = matching[0]
                result += [choice]
                print(f"{os.path.basename(choice)} added to {series_type} list.")
            else:
                result += matching
                print(f"{len(matching)} scans added to list.")
        else:
            print(f"Could not find {series_type} scan for subject {subject_id}!")
    print(f"Found {len(result)} {series_type} scans from {len(subjects)} subjects.")
    return result
