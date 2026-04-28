import re
import os

model_type = ("ResNet") #"DenseNet", "VGG", "Inception3")
model_path = ("/c/Users/Intel User/Desktop/blackbird_scripts")
dataset_path = ("/f/Stacked/Aug_2_22_Cas-adapt") # Path to the dataset
epoch = ("44")
up_threshold = ("0.95")
down_threshold = ("0.3")
outdim = ("2")
means = ("0.5663 0.6596 0.4508")
stds = ("0.1811 0.1667 0.2434")
spor_th = ("0.5")
inf_gate = ("0.3")
timestamp = ("Jan26_23-15-35_2026")

"""
This function generates the shell script commands for the a given base folder.

To run the function, you need to provide the base folder path and customize any variables below. The script will 
return a list of shell script commands for which you can save the output to a shell file.

"""

def generate_script_commands(base_folder):
    commands = []
    SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information", ".git", "__pycache__", "thumbs", "cache"}

    for root, dirs, files in os.walk(base_folder):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        # Check if there are PNG files in the current directory
        if any(file.endswith('.png') for file in files):
            #print("Found PNG files in", root)
            # Split the path to get each component
            path_parts = re.split(r'[\\/]', root)

            # Debugging print
            #print("Path parts:", path_parts)

            image_folder = None
            tray = None
            dpi = None

            # Check if the path has at least 2 parts
            if len(path_parts) >= 2:
                # Get the image folder and tray from the path
                img_folder = path_parts[-2]
                tray = path_parts[-1]

                # Debugging print
                #print("Image Folder:", img_folder, "Tray:", tray)

                # Extract dpi from the folder name (assuming it ends with '_<dpi>dpi')

                if 'dpi' in img_folder:
                    #print("True")
                    dpi = ''.join(re.findall(r'\d+', img_folder.split('_')[-1]))
                    #print("DPI:", dpi)
                    # Construct the command
                    cmd = (
                        "time python ../leaf_correlation_mw.py"
                        f" --model_type {model_type}"
                        f" --model_path {model_path}"
                        f" --dataset_path {dataset_path}"
                        f" --loading_epoch {epoch}"
                        f" --up_threshold {up_threshold}"
                        f" --down_threshold {down_threshold}"
                        f" --cuda"
                        f" --cuda_id 0"
                        f" --outdim {outdim}"
                        f" --means {means}"
                        f" --stds {stds}"
                        f" --timestamp {timestamp}"
                        f" --dpi {dpi}"
                        " --pretrained"
                        " --sal_thresh_method fixed"
                        " --sal_gradcam"
                        " --pm HPM-1269"
                        " --sal_smoothgrad"
                        " --dual_head"
                        f" --img_folder {img_folder}"
                        f" --trays {tray}"
                        f" --inf_gate {inf_gate}"
                        f" --spor_th {spor_th}"
                        f" --inf_gate {inf_gate}"
                    )

                    commands.append(cmd)
                    #print("Command generated:", cmd)

    return commands


if __name__ == "__main__":
    commands = generate_script_commands(r"F:/Stacked/Aug_2_22_Cas-adapt")
    for command in commands:
        print(f"\"{command}\"")
