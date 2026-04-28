import h5py
import numpy as np

# Open the HDF5 file
with h5py.File('mapping_population_dataset_6_28_23.hdf5', 'r') as f:

    # Get the number of samples
    num_samples = f['images'].shape[0] 

    # Shuffle the indices of the samples
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Split the indices into training and validation sets
    num_train = int(num_samples * 0.8)
    num_val = int(num_samples * 0.2)
    indices_train = sorted(indices[:num_train])
    indices_val_1 = sorted(indices[num_train:num_train+num_val])
    #indices_val_2 = sorted(indices[num_train+num_val:num_train+2*num_val])
    #indices_val_3 = sorted(indices[num_train+2*num_val:num_train+3*num_val])
    #indices_val_4 = sorted(indices[num_train+3*num_val:])

    # Create the datasets for the training and validation sets
    train_data = f['images'][indices_train]
    train_labels = f['labels'][indices_train]
    #train_masks = f['masks'][indices_train]
    val_data_1 = f['images'][indices_val_1]
    val_labels_1 = f['labels'][indices_val_1]
    #val_masks_1 = f['masks'][indices_val_1]
    #val_data_2 = f['images'][indices_val_2]
    #val_labels_2 = f['labels'][indices_val_2]
    #val_masks_2 = f['masks'][indices_val_2]
    #val_data_3 = f['images'][indices_val_3]
    #val_labels_3 = f['labels'][indices_val_3]
    #val_masks_3 = f['masks'][indices_val_3]
    #val_data_4 = f['images'][indices_val_4]
    #val_labels_4 = f['labels'][indices_val_4]
    #val_masks_4 = f['masks'][indices_val_4]

    # Save the datasets to new HDF5 files
    with h5py.File('train_mapping_pop_june_28_23.hdf5', 'w') as train_f:
        train_f.create_dataset('images', data=train_data)
        train_f.create_dataset('labels', data=train_labels)
        #train_f.create_dataset('masks', data=train_masks)

    with h5py.File('test_mapping_pop_june_28_23.hdf5', 'w') as val_f:
        val_f.create_dataset('images', data=val_data_1)
        val_f.create_dataset('labels', data=val_labels_1)
        #val_f.create_dataset('masks', data=val_masks_1)

    #with h5py.File('val_2.hdf5', 'w') as val_f:
        #val_f.create_dataset('images', data=val_data_2)
        #val_f.create_dataset('labels', data=val_labels_2)
        #val_f.create_dataset('masks', data=val_masks_2)

    #with h5py.File('val_3.hdf5', 'w') as val_f:
        #val_f.create_dataset('images', data=val_data_3)
        #val_f.create_dataset('labels', data=val_labels_3)
        #val_f.create_dataset('masks', data=val_masks_3)

    #with h5py.File('test.hdf5', 'w') as val_f:
        #val_f.create_dataset('images', data=val_data_4)
        #val_f.create_dataset('labels', data=val_labels_4)
        #val_f.create_dataset('masks', data=val_masks_4)
