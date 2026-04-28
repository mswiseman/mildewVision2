import h5py
import numpy as np


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))


def count_data_types_and_labels(file_path):
    data_info = {}
    label_counts = {0: 0, 1: 0, 2: 0}

    with h5py.File(file_path, 'r') as f:
        print("\n=== HDF5 structure and attributes ===")
        f.visititems(print_attrs)

        for key in f.keys():
            data = f[key]

            # Count data types and shapes
            data_type = str(data.dtype)
            data_shape = data.shape
            data_key = (data_type, data_shape)
            data_info[data_key] = data_info.get(data_key, 0) + 1

            print(f"\nPreview of '{key}':")
            preview_n = min(data.shape[0], 10) if data.ndim > 0 else 1
            print(np.array(data[:preview_n]))

            # ---- Label counting ----
            if key.lower() == "labels" or "label" in key.lower():
                arr = np.array(data)

                # Handle (N,1) by flattening
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr.reshape(-1)

                # Integer labels (N,)
                if arr.ndim == 1:
                    unique, counts = np.unique(arr, return_counts=True)
                    for u, c in zip(unique, counts):
                        u = int(u)
                        if u in label_counts:
                            label_counts[u] += int(c)

                # One-hot / probs (N,C)
                elif arr.ndim == 2 and arr.shape[1] >= 3:
                    labels = np.argmax(arr, axis=1)
                    unique, counts = np.unique(labels, return_counts=True)
                    for u, c in zip(unique, counts):
                        u = int(u)
                        if u in label_counts:
                            label_counts[u] += int(c)

    print("\n=== Data types, dimensions, and counts ===")
    for (dt, shape), count in data_info.items():
        print(f"{dt} of shape {shape}: {count}")

    print("\n=== Label counts ===")
    for k in sorted(label_counts):
        print(f"Label {k}: {label_counts[k]}")


file_path = r'C:\Users\Intel User\Desktop\blackbird_scripts\data\segmentation\train_set.hdf5'
count_data_types_and_labels(file_path)
