import os
import subprocess
import tempfile
import shutil
import concurrent.futures
import argparse

# Arg parser setup
def parse_args():
    parser = argparse.ArgumentParser(description="Batch process image stacks with Helicon Focus.")
    parser.add_argument('--input_dir', type=str, default="D:/Unstacked/",
                        help='Directory containing unstacked images')
    parser.add_argument('--output_dir_base', type=str, default="D:/Stacked/",
                        help='Base directory for stacked output images')
    parser.add_argument('--helicon_focus_path', type=str, default="C:/Program Files/Helicon Software/Helicon Focus 8/HeliconFocus.exe",
                        help='Path to Helicon Focus executable')
    parser.add_argument('--nconvert_path', type=str, default="C:/XnView/nconvert.exe",
                        help='Path to NConvert executable')
    parser.add_argument('--backup_dir_base', type=str, default="D:/Backup_Permanent",
                        help='Base directory for backup of unstacked files')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Number of parallel processes to run')
    parser.add_argument('--do_backup', action='store_true',
                        help='Backup unstacked files after processing')
    return parser.parse_args()


def process_stack_group(input_dir, output_dir_base, helicon_focus_path, nconvert_path,
                        backup_dir_base, experiment, date, tray, stack_group, do_backup):
    """
    Process a single stack group: stack images, convert to PNG, and handle backups.
    """
    tray_path = os.path.join(input_dir, experiment, date, tray)
    stack_group_path = os.path.join(tray_path, stack_group)

    output_dir = os.path.join(output_dir_base, experiment, date, tray)
    os.makedirs(output_dir, exist_ok=True)

    source_files = [os.path.join(stack_group_path, f)
                    for f in os.listdir(stack_group_path)
                    if f.lower().endswith('.nef')]

    if not source_files:
        print(f"Skipping empty stack group: {stack_group_path}")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix='.lst') as f:
        temp_file_path = f.name
        f.write('\n'.join(source_files).encode())

    helicon_output_file_path = os.path.join(output_dir, f'{stack_group}.tiff')

    subprocess.run([
        helicon_focus_path, '-silent', '-i', temp_file_path,
        f'-save:{helicon_output_file_path}', '-mp:1', '-rp:4', '-sp:2', '-tif:u'
    ], check=True)

    png_output_file_path = os.path.join(output_dir, f'{stack_group.rsplit("_", 1)[0]}.png')

    subprocess.run([
        nconvert_path, '-out', 'png', '-o', png_output_file_path, helicon_output_file_path
    ], check=True)

    os.remove(helicon_output_file_path)

    if do_backup:
        backup_dir = os.path.join(backup_dir_base, experiment, date, tray)
        os.makedirs(backup_dir, exist_ok=True)
        shutil.move(stack_group_path, os.path.join(backup_dir, stack_group))
    else:
        shutil.rmtree(stack_group_path)


def process_params(params):
    return process_stack_group(*params)


def main():
    args = parse_args()

    print("\n=== Beginning stacking ===")
    print(f"Using max workers: {args.max_workers}")
    print(f"Backing up files: {'Yes' if args.do_backup else 'No'}")
    print("............................")

    tasks = []
    for experiment in os.listdir(args.input_dir):
        exp_path = os.path.join(args.input_dir, experiment)
        if not os.path.isdir(exp_path):
            continue
        for date in os.listdir(exp_path):
            date_path = os.path.join(exp_path, date)
            if not os.path.isdir(date_path):
                continue
            for tray in os.listdir(date_path):
                tray_path = os.path.join(date_path, tray)
                if not os.path.isdir(tray_path):
                    continue
                for stack_group in os.listdir(tray_path):
                    group_path = os.path.join(tray_path, stack_group)
                    if os.path.isdir(group_path):
                        tasks.append((
                            args.input_dir, args.output_dir_base,
                            args.helicon_focus_path, args.nconvert_path,
                            args.backup_dir_base, experiment, date, tray,
                            stack_group, args.do_backup
                        ))

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        list(executor.map(process_params, tasks))

    print("\n=== Processing complete ===")


if __name__ == '__main__':
    main()
