import glob
import os
import pydicom
import subprocess

DATA_DIR = "/export/home/zvibaratz/Data/MRI"
OUTPUT = "/export/home/zvibaratz/Projects/reid/Scans"
DCM2NII = "/export/home/zvibaratz/Programs/MRIcroGL/dcm2niix"


def to_nifti(data_dir=DATA_DIR):
    series_dirs = glob.glob(os.path.join(data_dir, "*/*/"))
    for series in series_dirs:
        sample_dcm = glob.glob(os.path.join(series, "*.dcm"))[0]
        dcm = pydicom.dcmread(sample_dcm, stop_before_pixels=True)
        subject_id = dcm.PatientID.zfill(9)
        desc = dcm.SeriesDescription
        series_date = dcm.SeriesDate
        dest = os.path.join(OUTPUT, subject_id, series_date)
        os.makedirs(dest, exist_ok=True)
        command = [DCM2NII, "-z", "y", "-b", "n", "-o", dest, "-f", desc, series]
        subprocess.check_output(command)
