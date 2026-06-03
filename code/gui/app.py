import sys
import subprocess
import re
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
    QRadioButton,
    QButtonGroup,
    QGroupBox,
    QMessageBox,
)
import shutil

conda_exe = shutil.which("conda")

DEFAULT_CORRELATION_SCRIPT = "../leaf_correlation_mw.py"
DEFAULT_SALIENCY_PLOT_SCRIPT = "../plot_sal_map_leaf.py"
DEFAULT_CONDA_ENV = "mildewVision"
DEFAULT_CONDA_ENV_PYTHON = "C:/Users/Intel User/.conda/envs/mildewVision/python.exe"

PIPELINE_PRESETS = {
    "powdery": {
        "model_type": "ResNet",
        "model_path": "C:/Users/blackbird_user/Desktop/mildewVision2",
        "loading_epoch": 44,
        "up_threshold": 0.95,
        "down_threshold": 0.3,
        "cuda": True,
        "cuda_id": "0",
        "outdim": 2,
        "means": [0.5663, 0.6596, 0.4508],
        "stds": [0.1811, 0.1667, 0.2434],
        "timestamp": "Jan26_23-15-35_2026",
        "pretrained": True,
        "sal_thresh_method": "fixed",
        "dual_head": True,
        "inf_gate": 0.3,
        "spor_th": 0.5,
        "sal_gradcam": True,
        "sal_gradient": False,
        "sal_smoothgrad": True,
        "sal_deeplift": False,
        "store_both_sal_heads": False,
    },
    "downy": {
        "model_type": "VGG",
        "model_path": "C:/Users/blackbird_user/Desktop/mildewVision2",
        "loading_epoch": 66,
        "up_threshold": 0.8,
        "down_threshold": 0.2,
        "cuda": True,
        "cuda_id": "0",
        "outdim": 2,
        "means": [0.5765, 0.6403, 0.4478],
        "stds": [0.1584, 0.1574, 0.1902],
        "timestamp": "Nov26_02-59-03_2024",
        "pretrained": True,
        "sal_thresh_method": "fixed",
        "dual_head": False,
        "inf_gate": None,
        "spor_th": None,
        "sal_gradcam": True,
        "sal_gradient": False,
        "sal_smoothgrad": True,
        "sal_deeplift": False,
        "store_both_sal_heads": False,
    },
}

PARAMETER_HELP = {
    "model_type": (
        "Model type",
        "The neural network architecture used for inference.\n\n"
        "Powdery mildew usually uses ResNet.\n"
        "Downy mildew usually uses VGG."
    ),
    "model_path": (
        "Blackbird folder root path",
        "The root folder where blackbird code, logs, etc. subdirectories live.\n\n"
    ),
    "dataset_path": (
        "Dataset root path",
        "The root folder containing your image folders.\n\n"
        "Expected structure:\n"
        "dataset_path/image_folder/tray_folder/images.png"
    ),
    "loading_epoch": (
        "Loading epoch",
        "The training epoch checkpoint to load.\n\n"
        "For example, epoch 44 searches for files like:\n"
        "ResNet_model_ep044, ResNet_model_ep044.tar, .pth, or .pt"
    ),
    "timestamp": (
        "Model timestamp",
        "The timestamp string used in the trained model folder name.\n\n"
        "Example:\n"
        "Jan26_23-15-35_2026"
    ),
    "outdim": (
        "Output classes / outdim",
        "Number of output logits produced by the model.\n\n"
        "For your powdery dual-head model, this is usually 2."
    ),
    "pretrained": (
        "Pretrained",
        "Whether the model architecture was initialized with pretrained weights during training.\n\n"
        "This should match how the model was trained."
    ),
    "dual_head": (
        "Dual-head model",
        "Use this for the powdery mildew model with two output heads:\n\n"
        "Head 0: infected / hyphal signal\n"
        "Head 1: sporulation signal\n\n"
        "Usually checked for powdery mildew and unchecked for downy mildew."
    ),
    "cuda": (
        "Use CUDA",
        "Run inference on an CUDA-endabled GPU if available.\n\n"
    ),
    "cuda_id": (
        "CUDA ID",
        "Which GPU to use.\n\n"
        "Usually 0 if the computer has one GPU."
    ),
    "mps": (
        "Use MPS",
        "Use Apple Silicon GPU acceleration.\n\n"
        "Leave unchecked on your Windows/NVIDIA machine."
    ),
    "means": (
        "Channel means",
        "RGB normalization means used during model training.\n\n"
        "These must match the model being loaded."
    ),
    "stds": (
        "Channel stds",
        "RGB normalization standard deviations used during model training.\n\n"
        "These must match the model being loaded."
    ),
    "img_folder": (
        "Image folder",
        "The imaging-date folder inside the dataset root.\n\n"
        "Example:\n"
        "5-13-2026_5dpi"
    ),
    "dpi": (
        "DPI",
        "Days post inoculation.\n\n"
        "This can be inferred from image folder names ending in _5dpi, _10dpi, etc."
    ),
    "trays": (
        "Tray",
        "The tray folder inside the selected image folder.\n\n"
        "Example:\n"
        "HPM-663_T1 or tray_1"
    ),
    "pm": (
        "PM isolate / metadata",
        "Powdery mildew isolate name stored in the output CSV metadata.\n\n"
        "Example:\n"
        "HPM-663"
    ),
    "platform": (
        "Platform",
        "Imaging platform metadata.\n\n"
        "Usually BlackBird."
    ),
    "group": (
        "Group",
        "Experimental group metadata written to logs/outputs.\n\n"
        "Usually baseline unless you are separating runs."
    ),
    "step_size": (
        "Step size",
        "Sliding-window step size in pixels.\n\n"
        "Usually 224 for non-overlapping 224 x 224 patches."
    ),
    "up_threshold": (
        "Up threshold / infected threshold",
        "Patch-level infected threshold.\n\n"
        "If infected probability is greater than or equal to this value, the patch is called infected."
    ),
    "down_threshold": (
        "Down threshold / healthy threshold",
        "Patch-level healthy threshold.\n\n"
        "If infected probability is less than or equal to this value, the patch is called clear."
    ),
    "inf_gate": (
        "Infected-head gate",
        "Minimum infected-head probability required before allowing a sporulation call.\n\n"
        "This is mainly used for the powdery dual-head model. The goal is to prevent erroneous sporulation calls in "
        "patches that have very low infected-head signal."
    ),
    "spor_th": (
        "Sporulation threshold",
        "Patch-level sporulation threshold for the powdery dual-head model.\n\n"
        "If sporulation probability is above this threshold and the infection gate is met, the patch can be called "
        "sporulating."
    ),
    "sal_threshold": (
        "Saliency threshold",
        "Fixed threshold used to convert saliency maps into binary saliency regions.\n"
        "Used when saliency threshold method is fixed."
    ),
    "sal_thresh_method": (
        "Saliency threshold method",
        "How saliency maps are thresholded.\n\n"
        "fixed: use a fixed numeric threshold.\n"
        "percentile: threshold each map by a percentile."
    ),
    "sal_thresh_p": (
        "Saliency percentile",
        "Percentile used when saliency threshold method is percentile.\n\n"
        "For example, 95 keeps the most salient 5 percent of pixels."
    ),
    "sal_gradcam": (
        "Grad-CAM",
        "Enables Grad-CAM saliency maps."
    ),
    "sal_gradient": (
        "Gradient",
        "Enables basic gradient saliency maps."
    ),
    "sal_smoothgrad": (
        "SmoothGrad",
        "Enables SmoothGrad saliency maps."
    ),
    "sal_deeplift": (
        "DeepLift",
        "Enables DeepLift saliency maps."
    ),
    "store_both_sal_heads": (
        "Store both saliency heads",
        "For dual-head powdery models, compute and store both infected-head and sporulation-head saliency when qualifying conditions are met."
    ),
}


class BatchRunner(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, commands, max_parallel=1):
        super().__init__()
        self.commands = commands
        self.max_parallel = max_parallel

        self.should_stop = False
        self.active_processes = []
        self.process_lock = threading.Lock()

    def stop(self):
        """
        Request all running jobs to stop and terminate active subprocesses.
        """
        self.should_stop = True
        self.log_signal.emit("\nSTOP requested. Terminating active run(s)...\n")

        with self.process_lock:
            for process in self.active_processes:
                if process.poll() is None:
                    try:
                        process.terminate()
                    except Exception as e:
                        self.log_signal.emit(f"Could not terminate process: {e}")

    def run_one_command(self, cmd, job_label):
        if self.should_stop:
            self.log_signal.emit(f"\nSkipping {job_label}; stop was requested.\n")
            return 1

        self.log_signal.emit(f"\n===== Starting {job_label} =====\n")
        self.log_signal.emit(" ".join(f'"{x}"' if " " in x else x for x in cmd) + "\n")

        process = None

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            with self.process_lock:
                self.active_processes.append(process)

            if process.stdout is not None:
                for line in process.stdout:
                    if self.should_stop:
                        break
                    self.log_signal.emit(f"[{job_label}] {line.rstrip()}")

            if self.should_stop and process.poll() is None:
                process.terminate()

            process.wait()

            if self.should_stop:
                self.log_signal.emit(f"\n===== Stopped {job_label} =====\n")
            else:
                self.log_signal.emit(
                    f"\n===== Finished {job_label} with exit code {process.returncode} =====\n"
                )

            return process.returncode

        except Exception as e:
            self.log_signal.emit(f"\n===== ERROR in {job_label}: {e} =====\n")
            return 1

        finally:
            if process is not None:
                with self.process_lock:
                    if process in self.active_processes:
                        self.active_processes.remove(process)

    def run(self):
        total = len(self.commands)
        self.log_signal.emit(
            f"Launching {total} job(s) with up to {self.max_parallel} in parallel.\n"
        )

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []

            for item in self.commands:
                if self.should_stop:
                    break

                cmd = item["cmd"]
                job_label = item["label"]

                futures.append(
                    executor.submit(self.run_one_command, cmd, job_label)
                )

            completed = 0

            for future in as_completed(futures):
                completed += 1

                try:
                    future.result()
                except Exception as e:
                    self.log_signal.emit(f"\nWorker error: {e}\n")

                self.log_signal.emit(
                    f"\nProgress: {completed}/{total} job(s) complete.\n"
                )

                if self.should_stop:
                    break

        if self.should_stop:
            self.log_signal.emit("\nBatch run stopped by user.\n")
        else:
            self.log_signal.emit("\nBatch run completed.\n")

        self.finished_signal.emit()

class PowderyMildewGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Blackbird Disease Inference GUI")

        main_layout = QVBoxLayout()

        # ------------------------------------------------------------
        # Header / attribution section
        # ------------------------------------------------------------
        header_group_box = QGroupBox()
        header_layout = QVBoxLayout()

        title_label = QLabel("mildewVision2 Inference GUI")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        credit_label = QLabel(
            "GUI made by Michele Wiseman, v1.0 release May 21st 2026<br>"
            "Hover over parameters for quick tips or refer to the "
            '<a href="https://github.com/mswiseman/mildewVision2">mildewVision2 GitHub page</a>.'
        )
        credit_label.setAlignment(Qt.AlignCenter)
        credit_label.setOpenExternalLinks(True)
        credit_label.setWordWrap(True)

        header_layout.addWidget(title_label)
        header_layout.addWidget(credit_label)

        header_group_box.setLayout(header_layout)
        main_layout.addWidget(header_group_box)

        # ------------------------------------------------------------
        # Disease preset section
        # ------------------------------------------------------------
        disease_group_box = QGroupBox("Disease model preset")
        disease_layout = QHBoxLayout()

        self.powdery_radio = QRadioButton("Powdery mildew")
        self.downy_radio = QRadioButton("Downy mildew")


        self.disease_group = QButtonGroup()
        self.disease_group.addButton(self.powdery_radio)
        self.disease_group.addButton(self.downy_radio)

        self.powdery_radio.setChecked(True)

        self.powdery_radio.toggled.connect(self.apply_selected_preset)
        self.downy_radio.toggled.connect(self.apply_selected_preset)

        disease_layout.addWidget(self.powdery_radio)
        disease_layout.addWidget(self.downy_radio)
        disease_group_box.setLayout(disease_layout)
        main_layout.addWidget(disease_group_box)

        # ------------------------------------------------------------
        # Output mode section
        # ------------------------------------------------------------
        output_mode_group_box = QGroupBox("Output mode")
        output_mode_layout = QHBoxLayout()

        self.save_saliency_plots = QCheckBox("Save saliency map plots")
        self.save_saliency_plots.setChecked(False)
        self.save_saliency_plots.setToolTip(
            "Switches the script to plot_sal_map_leaf.py so saliency map plot outputs are saved. This is much slower \n"
            "and usually not necessary unless you're having model issues or want visualization for a pub."
        )

        self.save_saliency_plots.toggled.connect(self.update_script_for_output_mode)

        output_mode_layout.addWidget(self.save_saliency_plots)
        output_mode_group_box.setLayout(output_mode_layout)
        main_layout.addWidget(output_mode_group_box)

        # ------------------------------------------------------------
        # Script section
        # ------------------------------------------------------------

        script_group_box = QGroupBox("Script / Python environment")
        script_layout = QGridLayout()

        self.conda_exe_path = QLineEdit("auto")
        self.conda_exe_path.setToolTip(
            "Path to conda.exe. Leave as 'auto' to let the GUI find conda."
        )

        conda_browse_button = QPushButton("Browse conda")
        conda_browse_button.clicked.connect(self.browse_conda_exe)

        script_layout.addWidget(QLabel("Conda executable"), 2, 0)
        script_layout.addWidget(self.conda_exe_path, 2, 1, 1, 2)
        script_layout.addWidget(conda_browse_button, 2, 3)

        self.script_path = QLineEdit(DEFAULT_CORRELATION_SCRIPT)
        self.script_path.setToolTip(
            "Python script to run. Usually leaf_correlation_mw.py or plot_sal_map_leaf.py."
        )

        script_browse_button = QPushButton("Browse")
        script_browse_button.clicked.connect(self.browse_script)

        # Row 0: script picker
        script_layout.addWidget(QLabel("Python script"), 0, 0)
        script_layout.addWidget(self.script_path, 0, 1)
        script_layout.addWidget(script_browse_button, 0, 2)

        # Row 1: conda environment toggle/name
        self.use_conda_env = QCheckBox("Run using mildewVision Python")
        self.use_conda_env.setChecked(True)

        self.conda_python_path = QLineEdit(DEFAULT_CONDA_ENV_PYTHON)

        python_browse_button = QPushButton("Browse")
        python_browse_button.clicked.connect(self.browse_conda_python)

        script_group_box.setLayout(script_layout)
        main_layout.addWidget(script_group_box)
        # ------------------------------------------------------------
        # Path section
        # ------------------------------------------------------------
        path_group_box = QGroupBox("Paths")
        path_layout = QGridLayout()

        self.model_path = QLineEdit()
        self.dataset_path = QLineEdit("../../data")
        self.log_path = QLineEdit("../../results/logs/inference_gui.log")

        model_browse_button = QPushButton("Browse")
        dataset_browse_button = QPushButton("Browse")
        log_browse_button = QPushButton("Browse")

        model_browse_button.clicked.connect(lambda: self.browse_folder(self.model_path))
        dataset_browse_button.clicked.connect(self.browse_dataset_folder)
        log_browse_button.clicked.connect(lambda: self.browse_save_file(self.log_path))

        path_layout.addWidget(QLabel("Blackbird folder root path"), 0, 0)
        path_layout.addWidget(self.model_path, 0, 1)
        path_layout.addWidget(model_browse_button, 0, 2)
        self.model_path.setToolTip("The root folder where blackbird code, logs, etc. subdirectories live.")


        path_layout.addWidget(QLabel("Dataset root path"), 1, 0)
        path_layout.addWidget(self.dataset_path, 1, 1)
        path_layout.addWidget(dataset_browse_button, 1, 2)
        self.dataset_path.setToolTip("The root folder containing your image folders. Expected structure:\n"
                                        "dataset_path/image_folder/tray_folder/images.png")

        path_layout.addWidget(QLabel("Log path"), 2, 0)
        path_layout.addWidget(self.log_path, 2, 1)
        path_layout.addWidget(log_browse_button, 2, 2)
        self.log_path.setToolTip(
            "The file where console output from the runs will be saved. This is useful for keeping a permanent record \n"
            "of the run output, especially if you are running multiple batches or want to refer back to the results \n"
            "later. Make sure to set this to a .log or .txt file. If the file already exists, new output will be \n"
            "appended to it rather than overwriting."
        )

        path_group_box.setLayout(path_layout)
        main_layout.addWidget(path_group_box)

        # ------------------------------------------------------------
        # Model settings section
        # ------------------------------------------------------------
        model_group_box = QGroupBox("Model settings")
        model_layout = QGridLayout()

        self.model_type = QComboBox()
        self.model_type.addItems(["ResNet", "VGG", "Inception3"])

        self.loading_epoch = QSpinBox()
        self.loading_epoch.setRange(0, 200)
        self.loading_epoch.setToolTip("Epoch number of the model checkpoint to load.")

        self.timestamp = QLineEdit()

        #self.outdim = QSpinBox()
        #self.outdim.setRange(1, 10)

        #self.pretrained = QCheckBox("Pretrained")
        self.dual_head = QCheckBox("Dual-head model")

        self.cuda = QCheckBox("Use CUDA")
        #self.cuda_id = QLineEdit("0")

        self.means = QLineEdit()
        self.stds = QLineEdit()

        model_layout.addWidget(QLabel("Model type"), 0, 0)
        model_layout.addWidget(self.model_type, 0, 1)
        self.model_type.setToolTip("The neural network architecture used for inference. The best current powdery "
        "mildew model uses ResNet; the best current downy mildew usually uses VGG (May 2026).")

        # self.add_help_button(model_layout, 0, 2, "model_type")

        model_layout.addWidget(QLabel("Loading epoch"), 1, 0)
        model_layout.addWidget(self.loading_epoch, 1, 1)

        self.add_help_button(model_layout, 1, 2, "loading_epoch")

        model_layout.addWidget(QLabel("Model timestamp"), 2, 0)
        model_layout.addWidget(self.timestamp, 2, 1)
        self.timestamp.setToolTip(
            "The timestamp string used in the trained model folder name.")

        self.add_help_button(model_layout, 2, 2, "timestamp")

        #model_layout.addWidget(QLabel("Output classes / outdim"), 3, 0)
        #model_layout.addWidget(self.outdim, 3, 1)
        #self.add_help_button(model_layout, 3, 2, "outdim")

        #model_layout.addWidget(self.pretrained, 4, 0)
        model_layout.addWidget(self.dual_head, 3, 1)
        self.dual_head.setToolTip("Use this for the powdery mildew model with two output heads: infected/hyphal signal and sporulation signal. \n Usually checked for powdery mildew and unchecked for downy mildew.")
        model_layout.addWidget(self.cuda, 3, 0)

        # model_layout.addWidget(QLabel("CUDA ID"), 5, 1)
        # model_layout.addWidget(self.cuda_id, 5, 2)
        # self.add_help_button(model_layout, 5, 3, "cuda_id")

        model_layout.addWidget(QLabel("Channel means"), 4, 0)
        model_layout.addWidget(self.means, 4, 1, 1, 2)
        self.means.setToolTip("RGB normalization means used during model training.\n Do not change unless using new model.")

        model_layout.addWidget(QLabel("Channel stds"), 5, 0)
        model_layout.addWidget(self.stds, 5, 1, 1, 2)
        self.stds.setToolTip(
            "RGB normalization std dev. used during model training.\n Do not change unless using new model.")

        model_group_box.setLayout(model_layout)
        main_layout.addWidget(model_group_box)

        # ------------------------------------------------------------
        # Dataset/run section
        # ------------------------------------------------------------
        run_group_box = QGroupBox("Dataset / run settings")
        run_layout = QGridLayout()

        self.img_folder = QComboBox()
        self.img_folder.currentTextChanged.connect(self.on_image_folder_changed)

        self.dpi = QSpinBox()
        self.dpi.setRange(0, 100)
        self.dpi.setValue(10)
        self.dpi.setToolTip(
        "Days post inoculation. This value plays a role in the powdery mildew model logic for determining sporulation \n"
        "calls, so make sure it's correct. If your image folders are named with the dpi at the end like 5-13-2026_5dpi, \n"
        " 6-28-2023_10dpi, etc., selecting the folder will automatically set the dpi value. If not, set it manually here.")
        #self.add_help_button(run_layout, 1, 2, "dpi")

        self.trays = QComboBox()
        self.trays.setToolTip(
        "The tray folder inside the selected image folder. If your dataset is organized with trays inside image \n"
        "folders, select the appropriate tray here. If you want to run all trays within the selected image folder, \n"
        "check the 'Run all jobs' box and leave this dropdown at its default value.")

            #self.add_help_button(run_layout, 2, 2, "trays")

        self.pm = QLineEdit("")
        # self.platform = QLineEdit("BlackBird")
        self.group = QLineEdit("baseline")

        # self.step_size = QSpinBox()
        # self.step_size.setRange(1, 5000)
        # self.step_size.setValue(224)
        # self.add_help_button(run_layout, 5, 2, "step_size")

        refresh_data_button = QPushButton("Refresh folders")
        refresh_data_button.clicked.connect(self.refresh_image_folders)

        run_layout.addWidget(QLabel("Image folder"), 0, 0)
        run_layout.addWidget(self.img_folder, 0, 1)
        run_layout.addWidget(refresh_data_button, 0, 2)
        self.add_help_button(run_layout, 0, 3, "img_folder")
        self.img_folder.setToolTip(
        "The imaging-date folder inside the dataset root. For example, 5-13-2026_5dpi. The image folders are populated \n"
        "from the dataset root path you set in the Paths section. If your dataset is organized with trays inside image \n"
        "folders, select the appropriate tray here. If you want to run all trays within the selected image folder, \n"
        "check the 'Run all jobs' box and leave the tray dropdown at its default value.")

        run_layout.addWidget(QLabel("DPI"), 1, 0)
        run_layout.addWidget(self.dpi, 1, 1)

        run_layout.addWidget(QLabel("Tray"), 2, 0)
        run_layout.addWidget(self.trays, 2, 1)

        self.run_all_jobs = QCheckBox("Run all image folders and trays within data path")

        self.max_parallel_jobs = QSpinBox()
        self.max_parallel_jobs.setRange(1, 4)
        self.max_parallel_jobs.setValue(1)

        run_layout.addWidget(self.run_all_jobs, 3, 0, 1, 2)
        run_layout.addWidget(QLabel("Max parallel jobs"), 4, 0)
        run_layout.addWidget(self.max_parallel_jobs, 4, 1)
        self.max_parallel_jobs.setToolTip(
            "If running all image folders and trays, this controls how many run scripts are executed at the same \n"
            "time. Set to 1 for sequential runs, or higher to run multiple at once if you have the resources. Note \n"
            "that bigger models, such as VGG16, require more GPU memory and may not run successfully with multiple \n"
            "parallel jobs. Monitor GPU usage and adjust as needed. If you get out-of-memory errors, reduce this \n"
            "number or switch to sequential runs. The default max is set to 4."
        )

        run_layout.addWidget(QLabel("Isolate metadata"), 5, 0)
        run_layout.addWidget(self.pm, 5, 1)
        self.pm.setToolTip(
        "Optional field to keep track of powdery or downy mildew isolate identifiers. Be careful filling this out \n" 
        "when running batch runs across multiple isolates, as the value will be applied to all runs in the batch. \n"
        "You can also leave it blank if you don't need this metadata.")

        # run_layout.addWidget(QLabel("Platform"), 4, 0)
        # run_layout.addWidget(self.platform, 4, 1)

        run_layout.addWidget(QLabel("Group"), 6, 0)
        run_layout.addWidget(self.group, 6, 1)
        self.group.setToolTip(""
        "Optional metadata field to differentiate experimental groups. This is especially useful if you are running \n"
        " multiple batches of runs and want to keep their outputs organized in the logs and output files. You can set \n"
        " it to something like 'baseline', 'treatmentA', 'treatmentB', etc., or leave it as the default 'baseline'.")

        # run_layout.addWidget(QLabel("Step size"), 5, 0)
        # run_layout.addWidget(self.step_size, 5, 1)

        run_group_box.setLayout(run_layout)
        main_layout.addWidget(run_group_box)

        # ------------------------------------------------------------
        # Threshold section
        # ------------------------------------------------------------
        threshold_group_box = QGroupBox("Thresholds")
        threshold_layout = QGridLayout()

        self.up_threshold = self.double_box(0.95, 0.0, 1.0)
        self.down_threshold = self.double_box(0.30, 0.0, 1.0)
        self.inf_gate = self.double_box(0.30, 0.0, 1.0)
        self.spor_th = self.double_box(0.50, 0.0, 1.0)
        self.sal_threshold = self.double_box(0.50, 0.0, 1.0)

        self.sal_thresh_p = self.double_box(95.0, 0.0, 100.0)

        self.sal_thresh_method = QComboBox()
        self.sal_thresh_method.addItems(["fixed", "percentile"])

        self.allow_threshold_editing = QCheckBox("Allow threshold editing")
        self.allow_threshold_editing.setChecked(False)
        self.allow_threshold_editing.toggled.connect(self.set_threshold_editing_enabled)
        self.sal_threshold.setToolTip(
            "Uncheck if you want to alter the optimized threshold values. Note: the downy model doesn't current use \n"
            "the infection gate or sporulation threshold so these will be ignored if you have the downy preset selected."
        )

        threshold_layout.addWidget(self.allow_threshold_editing, 0, 0, 1, 2)

        threshold_layout.addWidget(QLabel("Up threshold / infected threshold"), 1, 0)
        threshold_layout.addWidget(self.up_threshold, 1, 1)
        self.up_threshold.setToolTip("Probability cutoff for calling a patch infected.")

        threshold_layout.addWidget(QLabel("Down threshold / healthy threshold"), 2, 0)
        threshold_layout.addWidget(self.down_threshold, 2, 1)
        self.down_threshold.setToolTip("Probability cutoff for calling a patch clear.")

        threshold_layout.addWidget(QLabel("Infection gate"), 3, 0)
        threshold_layout.addWidget(self.inf_gate, 3, 1)
        self.inf_gate.setToolTip(
            "Minimum infected-head probability required before a patch can be called sporulating."
        )

        threshold_layout.addWidget(QLabel("Sporulation threshold"), 4, 0)
        threshold_layout.addWidget(self.spor_th, 4, 1)
        self.spor_th.setToolTip("Sporulation-head cutoff for dual-head powdery mildew models.")

        threshold_layout.addWidget(QLabel("Saliency threshold"), 5, 0)
        threshold_layout.addWidget(self.sal_threshold, 5, 1)
        self.sal_threshold.setToolTip(
            "Fixed saliency cutoff used to convert saliency heatmaps into binary patch class-associated regions."
        )


        threshold_layout.addWidget(QLabel("Saliency threshold method"), 6, 0)
        threshold_layout.addWidget(self.sal_thresh_method, 6, 1)

        threshold_layout.addWidget(QLabel("Saliency percentile"), 7, 0)
        threshold_layout.addWidget(self.sal_thresh_p, 7, 1)

        threshold_group_box.setLayout(threshold_layout)
        main_layout.addWidget(threshold_group_box)

        # ------------------------------------------------------------
        # Saliency section
        # ------------------------------------------------------------
        saliency_group_box = QGroupBox("Saliency options")
        saliency_layout = QGridLayout()

        self.sal_gradcam = QCheckBox("Grad-CAM")
        self.sal_gradient = QCheckBox("Gradient")
        self.sal_smoothgrad = QCheckBox("SmoothGrad")
        self.sal_deeplift = QCheckBox("DeepLift")
        self.store_both_sal_heads = QCheckBox("Store both saliency heads")
        self.store_both_sal_heads.setToolTip(
            "For dual-head models, save saliency maps for both infected and sporulation heads when applicable.\n"
            "Leave this unchecked for standard runs. Turn it on when you want to compare where the infected-head and \n"
            " sporulation-head saliency maps overlap or differ."
        )

        saliency_layout.addWidget(self.sal_gradcam, 0, 0)
        saliency_layout.addWidget(self.sal_gradient, 0, 1)
        saliency_layout.addWidget(self.sal_smoothgrad, 1, 0)
        saliency_layout.addWidget(self.sal_deeplift, 1, 1)
        saliency_layout.addWidget(self.store_both_sal_heads, 2, 0, 1, 2)

        saliency_group_box.setLayout(saliency_layout)
        main_layout.addWidget(saliency_group_box)

        # ------------------------------------------------------------
        # Action buttons
        # ------------------------------------------------------------

        button_layout = QHBoxLayout()

        preview_button = QPushButton("Preview command")
        copy_button = QPushButton("Copy command")
        self.run_button = QPushButton("Run pipeline")
        self.stop_button = QPushButton("Stop all runs")
        self.stop_button.setEnabled(False)

        preview_button.clicked.connect(self.preview_command)
        copy_button.clicked.connect(self.copy_command)
        self.run_button.clicked.connect(self.run_pipeline)
        self.stop_button.clicked.connect(self.stop_all_runs)

        button_layout.addWidget(preview_button)
        button_layout.addWidget(copy_button)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)

        main_layout.addLayout(button_layout)

        # ------------------------------------------------------------
        # Output window
        # ------------------------------------------------------------
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        main_layout.addWidget(QLabel("Command preview / output"))
        main_layout.addWidget(self.output)

        self.setLayout(main_layout)

        # Apply default powdery preset after all widgets exist
        self.apply_selected_preset()
        self.set_threshold_editing_enabled(False)

        # Try to populate image folders from default dataset path
        self.refresh_image_folders()

    # ------------------------------------------------------------
    # Widget helper methods
    # ------------------------------------------------------------
    def double_box(self, value, minimum, maximum):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(4)
        box.setSingleStep(0.01)
        box.setValue(value)
        return box

    def show_help(self, key):
        title, message = PARAMETER_HELP.get(
            key,
            ("Help", "No help text has been written for this parameter yet.")
        )

        QMessageBox.information(self, title, message)

    def add_help_button(self, layout, row, col, key):
        button = QPushButton("?")
        button.setFixedWidth(28)
        button.setToolTip("Click for help")
        button.clicked.connect(lambda: self.show_help(key))
        layout.addWidget(button, row, col)

    def browse_folder(self, target_line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            target_line_edit.setText(folder)

    def browse_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select dataset root folder")
        if folder:
            self.dataset_path.setText(folder)
            self.refresh_image_folders()

    def browse_conda_python(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Python executable",
            "",
            "Python executable (python.exe);;All files (*)",
        )
        if file_path:
            self.conda_python_path.setText(file_path)

    def browse_script(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Python script",
            "",
            "Python files (*.py);;All files (*)",
        )
        if file_path:
            self.script_path.setText(file_path)

    def browse_save_file(self, target_line_edit):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select log file",
            "",
            "Log files (*.log);;Text files (*.txt);;All files (*)",
        )
        if file_path:
            target_line_edit.setText(file_path)

    def find_conda_executable(self):
        """
        Try to find conda in a portable way across machines.
        """
        manual_path = self.conda_exe_path.text().strip()

        if manual_path and manual_path.lower() != "auto":
            manual_path = self.normalize_windows_path(manual_path)
            if Path(manual_path).exists():
                return manual_path
            raise FileNotFoundError(f"Conda executable does not exist: {manual_path}")

        conda_exe = shutil.which("conda")
        if conda_exe:
            return conda_exe

        user_home = Path.home()

        possible_paths = [
            user_home / "miniconda3" / "Scripts" / "conda.exe",
            user_home / "anaconda3" / "Scripts" / "conda.exe",
            user_home / "Miniconda3" / "Scripts" / "conda.exe",
            user_home / "Anaconda3" / "Scripts" / "conda.exe",
            Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
            Path("C:/ProgramData/anaconda3/Scripts/conda.exe"),
            Path("C:/ProgramData/Miniconda3/Scripts/conda.exe"),
            Path("C:/ProgramData/Anaconda3/Scripts/conda.exe"),
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        return None

    def browse_conda_exe(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select conda executable",
            "",
            "Conda executable (conda.exe);;All files (*)",
        )
        if file_path:
            self.conda_exe_path.setText(file_path)

    # ------------------------------------------------------------
    # Dataset folder scanning logic
    # ------------------------------------------------------------
    def refresh_image_folders(self):
        """
        Populate image_folder dropdown from dataset_path.

        Assumes:
            dataset_path/
                image_folder/
                    tray/
        """
        dataset_root = Path(self.dataset_path.text().strip())

        self.img_folder.blockSignals(True)
        self.img_folder.clear()
        self.trays.clear()

        if not dataset_root.exists() or not dataset_root.is_dir():
            self.img_folder.blockSignals(False)
            self.output.append(f"Dataset path does not exist: {dataset_root}")
            return

        image_folders = [
            p.name for p in dataset_root.iterdir()
            if p.is_dir() and not p.name.endswith("_masking")
        ]

        image_folders = sorted(image_folders)

        self.img_folder.addItems(image_folders)
        self.img_folder.blockSignals(False)

        if image_folders:
            self.img_folder.setCurrentIndex(0)
            self.refresh_trays()
            self.infer_dpi_from_image_folder()
        else:
            self.output.append(f"No image folders found in: {dataset_root}")

    def refresh_trays(self):
        """
        Populate tray dropdown from selected image_folder.
        Keeps exact folder names, e.g. tray_1, tray_2, 1, 2.
        """
        dataset_root = Path(self.dataset_path.text().strip())
        image_folder = self.img_folder.currentText().strip()

        image_folder_path = dataset_root / image_folder

        self.trays.clear()

        if not image_folder_path.exists() or not image_folder_path.is_dir():
            return

        tray_folders = [
            p.name for p in image_folder_path.iterdir()
            if p.is_dir()
        ]

        def tray_sort_key(name):
            numbers = re.findall(r"\d+", name)
            return int(numbers[0]) if numbers else name

        tray_folders = sorted(tray_folders, key=tray_sort_key)

        self.trays.addItems(tray_folders)

    def infer_dpi_from_image_folder(self):
        """
        Try to infer dpi from image folder names like:
            6-28-2023_10dpi
            4-1-2026_6dpi
            imagefolder_10dpi
        """
        image_folder = self.img_folder.currentText().strip()
        match = re.search(r"(\d+)\s*dpi", image_folder, re.IGNORECASE)

        if match:
            self.dpi.setValue(int(match.group(1)))

    def on_image_folder_changed(self):
        self.refresh_trays()
        self.infer_dpi_from_image_folder()

    def infer_dpi_value_from_name(self, image_folder_name):
        """
        Infer dpi from folder names ending like:
            5-18-2026_10dpi -> 10
            5-13-2026_5dpi  -> 5
        """
        match = re.search(r"_(\d+)\s*dpi$", image_folder_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return self.dpi.value()

    def normalize_windows_path(self, path_text):
        """
        Convert Git Bash / MSYS-style paths like:
            /c/Users/name/Desktop/project
        into Windows-friendly paths like:
            C:/Users/name/Desktop/project
        """
        path_text = path_text.strip()

        match = re.match(r"^/([a-zA-Z])/(.*)", path_text)
        if match:
            drive = match.group(1).upper()
            rest = match.group(2)
            return f"{drive}:/{rest}"

        return path_text

    def discover_all_image_tray_jobs(self):
        """
        Find all image_folder/tray combinations under dataset_path.

        Expected structure:
            dataset_path/
                image_folder/
                    tray_folder/
                        images.png
        """
        dataset_root = Path(self.normalize_windows_path(self.dataset_path.text()))

        jobs = []

        if not dataset_root.exists() or not dataset_root.is_dir():
            self.output.append(f"Dataset path does not exist: {dataset_root}")
            return jobs

        image_folders = [
            p for p in dataset_root.iterdir()
            if p.is_dir() and not p.name.endswith("_masking")
        ]

        image_folders = sorted(image_folders, key=lambda p: p.name)

        for image_folder_path in image_folders:
            tray_folders = [
                p for p in image_folder_path.iterdir()
                if p.is_dir()
            ]

            tray_folders = sorted(tray_folders, key=lambda p: p.name)

            for tray_path in tray_folders:
                png_files = list(tray_path.glob("*.png"))

                # Skip folders that do not contain images
                if not png_files:
                    continue

                jobs.append({
                    "image_folder": image_folder_path.name,
                    "tray": tray_path.name,
                    "dpi": self.infer_dpi_value_from_name(image_folder_path.name),
                })

        return jobs

    # ------------------------------------------------------------
    # Preset logic
    # ------------------------------------------------------------
    def apply_selected_preset(self):
        if not hasattr(self, "model_type"):
            return

        if self.powdery_radio.isChecked():
            preset = PIPELINE_PRESETS["powdery"]
        else:
            preset = PIPELINE_PRESETS["downy"]

        self.model_type.setCurrentText(preset["model_type"])
        self.model_path.setText(preset["model_path"])
        self.loading_epoch.setValue(preset["loading_epoch"])
        self.timestamp.setText(preset["timestamp"])
        #self.outdim.setValue(preset["outdim"])

        self.up_threshold.setValue(preset["up_threshold"])
        self.down_threshold.setValue(preset["down_threshold"])

        self.cuda.setChecked(preset["cuda"])
        # self.cuda_id.setText(str(preset["cuda_id"]))
        # self.pretrained.setChecked(preset["pretrained"])
        self.dual_head.setChecked(preset["dual_head"])

        self.means.setText(" ".join(str(x) for x in preset["means"]))
        self.stds.setText(" ".join(str(x) for x in preset["stds"]))

        self.sal_thresh_method.setCurrentText(preset["sal_thresh_method"])

        self.sal_gradcam.setChecked(preset["sal_gradcam"])
        self.sal_gradient.setChecked(preset["sal_gradient"])
        self.sal_smoothgrad.setChecked(preset["sal_smoothgrad"])
        self.sal_deeplift.setChecked(preset["sal_deeplift"])
        self.store_both_sal_heads.setChecked(preset["store_both_sal_heads"])

        if preset["inf_gate"] is not None:
            self.inf_gate.setValue(preset["inf_gate"])
            self.inf_gate.setEnabled(True)
        else:
            self.inf_gate.setEnabled(False)

        if preset["spor_th"] is not None:
            self.spor_th.setValue(preset["spor_th"])
            self.spor_th.setEnabled(True)
        else:
            self.spor_th.setEnabled(False)

        if self.downy_radio.isChecked():
            self.store_both_sal_heads.setEnabled(False)
        else:
            self.store_both_sal_heads.setEnabled(True)

        self.set_threshold_editing_enabled(self.allow_threshold_editing.isChecked())

    def set_threshold_editing_enabled(self, enabled):
        """
        Lock/unlock threshold-related settings.
        """
        threshold_widgets = [
            self.up_threshold,
            self.down_threshold,
            self.inf_gate,
            self.spor_th,
            self.sal_threshold,
            self.sal_thresh_method,
            self.sal_thresh_p,
        ]

        for widget in threshold_widgets:
            widget.setEnabled(enabled)

        # Preserve disease-specific logic:
        # If the selected preset is downy, keep powdery-only thresholds disabled.
        if self.downy_radio.isChecked():
            self.inf_gate.setEnabled(False)
            self.spor_th.setEnabled(False)

    # ------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------
    def parse_space_or_comma_list(self, text):
        return text.replace(",", " ").split()

    def update_script_for_output_mode(self, checked):
        """
        Switch default script depending on whether the user wants saliency map plots.
        """
        if checked:
            self.script_path.setText(DEFAULT_SALIENCY_PLOT_SCRIPT)
        else:
            self.script_path.setText(DEFAULT_CORRELATION_SCRIPT)

    def build_command_for_job(self, image_folder=None, tray=None, dpi=None):
        script = self.script_path.text().strip()
        means = self.parse_space_or_comma_list(self.means.text())
        stds = self.parse_space_or_comma_list(self.stds.text())

        selected_image_folder = image_folder or self.img_folder.currentText().strip()
        selected_tray = tray or self.trays.currentText().strip()
        selected_dpi = dpi if dpi is not None else self.dpi.value()

        if self.use_conda_env.isChecked():
            cmd_start = [
                self.normalize_windows_path(self.conda_python_path.text()),
                script,
            ]
        else:
            cmd_start = [
                sys.executable,
                script,
            ]

        cmd = cmd_start + [
            "--model_type", self.model_type.currentText(),
            "--model_path", self.normalize_windows_path(self.model_path.text()),
            "--dataset_path", self.normalize_windows_path(self.dataset_path.text()),
            "--loading_epoch", str(self.loading_epoch.value()),
            "--timestamp", self.timestamp.text().strip(),
            #"--outdim", str(self.outdim.value()),
            "--up_threshold", str(self.up_threshold.value()),
            "--down_threshold", str(self.down_threshold.value()),
            "--dpi", str(selected_dpi),
            "--img_folder", selected_image_folder,
            #"--platform", self.platform.text().strip(),
            "--group", self.group.text().strip(),
            #"--step_size", str(self.step_size.value()),
            "--sal_threshold", str(self.sal_threshold.value()),
            "--sal_thresh_method", self.sal_thresh_method.currentText(),
            "--sal_thresh_p", str(self.sal_thresh_p.value()),
            "--log", self.normalize_windows_path(self.log_path.text()),
            "--means", *means,
            "--stds", *stds,
        ]

        if selected_tray:
            cmd += ["--trays", selected_tray]

        pm_value = self.pm.text().strip()
        if pm_value:
            cmd += ["--pm", pm_value]

        if self.cuda.isChecked():
            cmd += ["--cuda"]

        #if self.mps.isChecked():
        #    cmd.append("--mps")

        #if self.pretrained.isChecked():
        #    cmd.append("--pretrained")

        if self.dual_head.isChecked():
            cmd.append("--dual_head")
            cmd += ["--inf_gate", str(self.inf_gate.value())]
            cmd += ["--spor_th", str(self.spor_th.value())]

        if self.sal_gradcam.isChecked():
            cmd.append("--sal_gradcam")

        if self.sal_gradient.isChecked():
            cmd.append("--sal_gradient")

        if self.sal_smoothgrad.isChecked():
            cmd.append("--sal_smoothgrad")

        if self.sal_deeplift.isChecked():
            cmd.append("--sal_deeplift")

        if self.store_both_sal_heads.isChecked() and self.dual_head.isChecked():
            cmd.append("--store_both_sal_heads")

        if self.use_conda_env.isChecked():
            conda_exe = self.find_conda_executable()

            if conda_exe is None:
                raise FileNotFoundError(
                    "Could not find conda. Either launch this GUI from an Anaconda/Miniconda prompt, "
                    "or add conda to PATH, or browse to conda.exe in the GUI."
                )

            cmd_start = [
                conda_exe,
                "run",
                "-n",
                self.conda_exe_path.text().strip(),
                "python",
                script,
            ]
        else:
            cmd_start = [
                sys.executable,
                script,
            ]

        return cmd

    def build_command(self):
        return self.build_command_for_job()

    def command_to_string(self, cmd):
        return " ".join(f'"{x}"' if " " in x else x for x in cmd)

    # ------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------
    def preview_command(self):
        self.output.clear()

        if self.run_all_jobs.isChecked():
            jobs = self.discover_all_image_tray_jobs()

            self.output.append(f"Batch mode: found {len(jobs)} job(s).\n")

            for job in jobs:
                cmd = self.build_command_for_job(
                    image_folder=job["image_folder"],
                    tray=job["tray"],
                    dpi=job["dpi"],
                )

                label = f'{job["image_folder"]} / {job["tray"]}'
                self.output.append(f"\n--- {label} ---")
                self.output.append(self.command_to_string(cmd))

        else:
            cmd = self.build_command()
            self.output.append("Command preview:\n")
            self.output.append(self.command_to_string(cmd))

    def copy_command(self):
        cmd = self.build_command()
        QApplication.clipboard().setText(self.command_to_string(cmd))
        self.output.append("\n\nCommand copied to clipboard.")

    def on_runner_finished(self):
        self.output.append("\nRun manager finished.")
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def stop_all_runs(self):
        """
        Stop any active single-run or batch-run subprocesses.
        """
        if hasattr(self, "batch_runner") and self.batch_runner is not None:
            if self.batch_runner.isRunning():
                self.batch_runner.stop()
                self.output.append("\nStop signal sent to active run(s).")
            else:
                self.output.append("\nNo active run is currently running.")
        else:
            self.output.append("\nNo active run is currently running.")

    def run_pipeline(self):
        self.output.clear()

        if self.run_all_jobs.isChecked():
            jobs = self.discover_all_image_tray_jobs()

            if not jobs:
                self.output.append("No jobs found. Check dataset folder structure.")
                return

            commands = []

            for job in jobs:
                cmd = self.build_command_for_job(
                    image_folder=job["image_folder"],
                    tray=job["tray"],
                    dpi=job["dpi"],
                )

                label = f'{job["image_folder"]}/{job["tray"]}'
                commands.append({
                    "cmd": cmd,
                    "label": label,
                })

            max_parallel = self.max_parallel_jobs.value()

            self.output.append(
                f"Starting batch run with {len(commands)} job(s), "
                f"up to {max_parallel} in parallel.\n"
            )

            self.batch_runner = BatchRunner(commands, max_parallel=max_parallel)
            self.batch_runner.log_signal.connect(self.output.append)
            self.batch_runner.finished_signal.connect(
                lambda: self.output.append("\nAll batch jobs finished.")
            )
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            self.batch_runner.start()
            self.batch_runner.finished_signal.connect(self.on_runner_finished)

        else:
            cmd = self.build_command()

            self.output.append("Running command:\n")
            self.output.append(self.command_to_string(cmd))
            self.output.append("\n\nOutput:\n")

            self.batch_runner = BatchRunner(
                [{"cmd": cmd, "label": "single run"}],
                max_parallel=1,
            )
            self.batch_runner.log_signal.connect(self.output.append)
            self.batch_runner.finished_signal.connect(
                lambda: self.output.append("\nRun finished.")
            )
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            self.batch_runner.start()
            self.batch_runner.finished_signal.connect(self.on_runner_finished)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PowderyMildewGUI()
    window.resize(1000, 1000)
    window.show()
    sys.exit(app.exec())
