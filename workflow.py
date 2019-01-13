from nipype import Node
from nipype.interfaces.fsl import BET

DATA_DIR = "/export/home/zvibaratz/Projects/reid/Skull-stripped"
TARGET = "/export/home/zvibaratz/Projects/reid/target/MPRAGE.nii.gz"
SERIES_TYPE = "t1"
COST_FUNCTION = "Mutual Information"
COST_FUNCTION_DICT = {
    "Mutual Information": "mutualinfo",
    "Correlation Ratio": "corratio",
    "Normalized Correlation": "normcorr",
    "Normalized Mutual Information": "normmi",
    "Least Squares": "leastsq",
    "Boundary-Based Registration": "bbr",
}

