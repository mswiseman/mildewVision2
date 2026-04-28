import os

path = r'G:\make_patches'

# add _clear to the end of all file name strings (before .png) within a given directory
for filename in os.listdir(path):
    if filename.endswith(".png"):
        name, ext = os.path.splitext(filename)
        new_filename = name + "_clear" + ext
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))