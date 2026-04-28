import os

dir2 = 'C:/Users/michele.wiseman/Desktop/Saliency_based_Grape_PM_Quantification-main/code/Pytorch-UNet/data/masks/'
dir1 = 'C:/Users/michele.wiseman/Desktop/Saliency_based_Grape_PM_Quantification-main/code/Pytorch-UNet/data/imgs/'

for filename in os.listdir(dir1):
    name, ext = os.path.splitext(filename)
    if not any(os.path.exists(os.path.join(dir2, name + ext)) for ext in ('.png', '.gif')):
        os.remove(os.path.join(dir1, filename))
        print("Deleted " + filename + " because it has no corresponding mask.")