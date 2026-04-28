import os
import pandas as pd

#path = r'C:\Users\michele.wiseman\Desktop\Saliency_based_Grape_PM_Quantification-main\data\labeled'
path = r'D:\mapping_population_patches_5dpi_model'


# Create a list to store the data
data = []

# Loop through all files in the directory
for filename in os.listdir(path):
    if filename.endswith('.png'):
        # Split the filename to get the index number
        index = int(filename.split('_')[0])
        # Check the label and assign a 0 for clear and a 1 for infected
        if 'clear' in filename:
            label = 0
        else:
            label = 1
        # Add the index and label to the data list
        data.append([index, label])

# Convert the data list to a Pandas DataFrame and save it as a csv
df = pd.DataFrame(data, columns=['Index', 'Label'])
df.to_csv('labels_hop_mapping_pop_patches_6_28_23.csv', index=False)


# after creating a labels file, you can then make an hdf5 file using h5py_creation.py
# make sure to delete the first row and first column of the csv file before running h5py_creation.py