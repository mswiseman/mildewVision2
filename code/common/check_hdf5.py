import h5py, numpy as np
from pathlib import Path

h5_path = Path("C:/Users/Intel User/Desktop/blackbird_scripts/data/train_1_26_26_3_class_powdery.hdf5")  # whatever you use
with h5py.File(h5_path, "r") as f:
    y = f["labels"][:]
print("Label counts after save:", dict(zip(*np.unique(y, return_counts=True))))
