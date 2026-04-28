import os

def list_directories(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            #if not f.endswith('.png'):
            print(f"{subindent}{f}")

# Replace 'your_directory_path' with the path of the directory you want to map
directory_path = r'E:/Stacked/Results/ResNet_upth0.8_downth0.3_Jan29_16-28-38_2024'
list_directories(directory_path)
