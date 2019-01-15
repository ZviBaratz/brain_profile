import glob
import os

from dao import LOCATION_DICT, get_scans
from nipype import Node
from nipype.interfaces.fsl import BET


def get_default_destination(scan: str, create: bool = True) -> str:
    output_dir = LOCATION_DICT["skull_stripped"]
    parts = scan.split("/")
    file_name, subject_id = parts[-1], parts[-2]
    if create:
        os.makedirs(os.path.join(output_dir, subject_id), exist_ok=True)
    return os.path.join(output_dir, subject_id, file_name)


def run_bet(robust: bool = True, skip_existing: bool = True):
    scans = get_scans(LOCATION_DICT["raw"])
    for scan in scans:
        print(f"\nCurrent series: {scan}")
        if skip_existing:
            print("Checking for existing skull-stripping output...", end="\t")
        dest = get_default_destination(scan)
        if skip_existing and os.path.isfile(dest):
            print(f"\u2714")
            continue
        print(f"\u2718")
        print("Running skull-stripping with BET...", end="\t")
        try:
            bet = BET(robust=True)
            bet.inputs.in_file = scan
            bet.inputs.out_file = dest
            bet.run()
            print(f"\u2714\tDone!")
        except Exception as e:
            print(f"\u2718")
            print(e.args)
            break
