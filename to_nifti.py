import glob
import os
import pydicom
import subprocess

DCM2NII = "/export/home/zvibaratz/Programs/MRIcroGL/dcm2niix"


def to_nifti(source_dir: str, dest_dir: str = None):
    series_dirs = glob.glob(os.path.join(source_dir, "*/*/"))
    if dest_dir is None:
        dest_dir = os.path.join(source_dir, "NIfTI")
        os.makedirs(dest_dir, exist_ok=True)
    for series in series_dirs:
        sample_dcm = glob.glob(os.path.join(series, "*.dcm"))[0]
        dcm = pydicom.dcmread(sample_dcm, stop_before_pixels=True)
        subject_id = dcm.PatientID.zfill(9)
        series_description = dcm.SeriesDescription
        if "IR-EPI" in series_description:
            file_name = "IR-EPI_TI" + dcm.InversionTime
        else:
            file_name = series_description
        series_date = dcm.SeriesDate
        dest = os.path.join(dest_dir, subject_id, series_date)
        os.makedirs(dest, exist_ok=True)
        command = [DCM2NII, "-z", "y", "-b", "n", "-o", dest, "-f", file_name, series]
        subprocess.check_output(command)
