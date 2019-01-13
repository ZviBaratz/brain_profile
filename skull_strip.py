import glob
import os

from nipype import Node
from nipype.interfaces.fsl import BET

DATA_DIR = '/export/home/zvibaratz/Projects/reid/Scans'
PATTERN = '**/*.nii.gz'
OUTPUT = '/export/home/zvibaratz/Projects/reid/Skull-stripped'


def get_default_destination(
        scan: str,
        create: bool = True,
) -> str:
    parts = scan.split('/')
    file_name, subject_id = parts[-1], parts[-2]
    if create:
        os.makedirs(os.path.join(OUTPUT, subject_id), exist_ok=True)
    return os.path.join(OUTPUT, subject_id, file_name)


def run_bet(
        robust: bool = True,
        skip_existing: bool = True,
):
    full_pattern = os.path.join(DATA_DIR, PATTERN)
    scans = glob.iglob(full_pattern, recursive=True)
    for scan in scans:
        print(f'\nCurrent series: {scan}')
        if skip_existing:
            print('Checking for existing skull-stripping output...', end='\t')
        dest = get_default_destination(scan)
        if skip_existing and os.path.isfile(dest):
            print(f'\u2714')
            continue
        print(f'\u2718')
        print('Running skull-stripping with BET...')
        try:
            bet = Node(BET(robust=True), name='bet_node')
            bet.inputs.in_file = scan
            bet.inputs.out_file = dest
            bet.run()
            print(f'\u2714\tDone!')
        except Exception as e:
            print(f'\u2718')
            print(e.args)
            break
