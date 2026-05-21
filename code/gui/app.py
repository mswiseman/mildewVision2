import sys
import subprocess
import re
from pathlib import Path

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
        "sal_thresh_method": "percentile",
        "dual_head": False,
        "inf_gate": None,
        "spor_th": None,
        "sal_gradcam": False,
        "sal_gradient": False,
        "sal_smoothgrad": False,
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
        "Model root path",
        "The root folder where the trained model results are stored.\n\n"
        "The script expects to find checkpoints under:\n"
        "model_path/results/models/model_name_timestamp/"
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
        "Run inference on an NVIDIA GPU.\n\n"
        "Use this on the 309 computer if PyTorch CUDA is available."
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
        "Infection gate",
        "Minimum infected-head probability required before allowing a sporulation call.\n\n"
        "This is mainly used for the powdery dual-head model."
    ),
    "spor_th": (
        "Sporulation threshold",
        "Patch-level sporulation threshold for the powdery dual-head model.\n\n"
        "If sporulation probability is above this threshold and the infection gate is met, the patch can be called sporulating."
    ),
    "sal_threshold": (
        "Saliency threshold",
        "Fixed threshold used to convert saliency maps into binary saliency regions.\n\n"
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

class PowderyMildewGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Blackbird Disease Inference GUI")

        main_layout = QVBoxLayout()

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
        # Script section
        # ------------------------------------------------------------
        script_group_box = QGroupBox("Script")
        script_layout = QGridLayout()

        self.script_path = QLineEdit("../leaf_correlation_mw.py")
        script_browse_button = QPushButton("Browse")
        script_browse_button.clicked.connect(self.browse_script)

        script_layout.addWidget(QLabel("Python script"), 0, 0)
        script_layout.addWidget(self.script_path, 0, 1)
        script_layout.addWidget(script_browse_button, 0, 2)

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

        path_layout.addWidget(QLabel("Model root path"), 0, 0)
        path_layout.addWidget(self.model_path, 0, 1)
        path_layout.addWidget(model_browse_button, 0, 2)

        path_layout.addWidget(QLabel("Dataset root path"), 1, 0)
        path_layout.addWidget(self.dataset_path, 1, 1)
        path_layout.addWidget(dataset_browse_button, 1, 2)

        path_layout.addWidget(QLabel("Log path"), 2, 0)
        path_layout.addWidget(self.log_path, 2, 1)
        path_layout.addWidget(log_browse_button, 2, 2)

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
        self.loading_epoch.setRange(0, 100000)

        self.timestamp = QLineEdit()

        self.outdim = QSpinBox()
        self.outdim.setRange(1, 10)

        self.pretrained = QCheckBox("Pretrained")
        self.dual_head = QCheckBox("Dual-head model")

        self.cuda = QCheckBox("Use CUDA")
        self.cuda_id = QLineEdit("0")

        self.means = QLineEdit()
        self.stds = QLineEdit()

        model_layout.addWidget(QLabel("Model type"), 0, 0)
        model_layout.addWidget(self.model_type, 0, 1)
        self.add_help_button(model_layout, 0, 2, "model_type")

        model_layout.addWidget(QLabel("Loading epoch"), 1, 0)
        model_layout.addWidget(self.loading_epoch, 1, 1)
        self.add_help_button(model_layout, 1, 2, "loading_epoch")

        model_layout.addWidget(QLabel("Model timestamp"), 2, 0)
        model_layout.addWidget(self.timestamp, 2, 1)
        self.add_help_button(model_layout, 2, 2, "timestamp")

        model_layout.addWidget(QLabel("Output classes / outdim"), 3, 0)
        model_layout.addWidget(self.outdim, 3, 1)
        self.add_help_button(model_layout, 3, 2, "outdim")

        model_layout.addWidget(self.pretrained, 4, 0)
        model_layout.addWidget(self.dual_head, 4, 1)
        self.add_help_button(model_layout, 4, 2, "dual_head")

        model_layout.addWidget(self.cuda, 5, 0)
        #model_layout.addWidget(QLabel("CUDA ID"), 5, 1)
        #model_layout.addWidget(self.cuda_id, 5, 2)
        #self.add_help_button(model_layout, 5, 3, "cuda_id")

        model_layout.addWidget(QLabel("Channel means"), 6, 0)
        model_layout.addWidget(self.means, 6, 1, 1, 2)
        self.add_help_button(model_layout, 6, 3, "means")

        model_layout.addWidget(QLabel("Channel stds"), 7, 0)
        model_layout.addWidget(self.stds, 7, 1, 1, 2)
        self.add_help_button(model_layout, 7, 3, "stds")

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
        self.add_help_button(run_layout, 1, 2, "dpi")

        self.trays = QComboBox()
        self.add_help_button(run_layout, 2, 2, "trays")

        self.pm = QLineEdit("")
        #self.platform = QLineEdit("BlackBird")
        self.group = QLineEdit("baseline")

        #self.step_size = QSpinBox()
        #self.step_size.setRange(1, 5000)
        #self.step_size.setValue(224)
        #self.add_help_button(run_layout, 5, 2, "step_size")

        refresh_data_button = QPushButton("Refresh folders")
        refresh_data_button.clicked.connect(self.refresh_image_folders)

        run_layout.addWidget(QLabel("Image folder"), 0, 0)
        run_layout.addWidget(self.img_folder, 0, 1)
        run_layout.addWidget(refresh_data_button, 0, 2)
        self.add_help_button(run_layout, 0, 3, "img_folder")

        run_layout.addWidget(QLabel("DPI"), 1, 0)
        run_layout.addWidget(self.dpi, 1, 1)

        run_layout.addWidget(QLabel("Tray"), 2, 0)
        run_layout.addWidget(self.trays, 2, 1)

        run_layout.addWidget(QLabel("PM isolate / metadata"), 3, 0)
        run_layout.addWidget(self.pm, 3, 1)

        #run_layout.addWidget(QLabel("Platform"), 4, 0)
        #run_layout.addWidget(self.platform, 4, 1)

        run_layout.addWidget(QLabel("Group"), 4, 0)
        run_layout.addWidget(self.group, 4, 1)

        #run_layout.addWidget(QLabel("Step size"), 5, 0)
        #run_layout.addWidget(self.step_size, 5, 1)

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

        threshold_layout.addWidget(QLabel("Up threshold / infected threshold"), 0, 0)
        threshold_layout.addWidget(self.up_threshold, 0, 1)

        threshold_layout.addWidget(QLabel("Down threshold / healthy threshold"), 1, 0)
        threshold_layout.addWidget(self.down_threshold, 1, 1)

        threshold_layout.addWidget(QLabel("Infection gate"), 2, 0)
        threshold_layout.addWidget(self.inf_gate, 2, 1)

        threshold_layout.addWidget(QLabel("Sporulation threshold"), 3, 0)
        threshold_layout.addWidget(self.spor_th, 3, 1)

        threshold_layout.addWidget(QLabel("Saliency threshold"), 4, 0)
        threshold_layout.addWidget(self.sal_threshold, 4, 1)

        threshold_layout.addWidget(QLabel("Saliency threshold method"), 5, 0)
        threshold_layout.addWidget(self.sal_thresh_method, 5, 1)

        threshold_layout.addWidget(QLabel("Saliency percentile"), 6, 0)
        threshold_layout.addWidget(self.sal_thresh_p, 6, 1)

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
        run_button = QPushButton("Run pipeline")

        preview_button.clicked.connect(self.preview_command)
        copy_button.clicked.connect(self.copy_command)
        run_button.clicked.connect(self.run_pipeline)

        button_layout.addWidget(preview_button)
        button_layout.addWidget(copy_button)
        button_layout.addWidget(run_button)

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
        self.outdim.setValue(preset["outdim"])

        self.up_threshold.setValue(preset["up_threshold"])
        self.down_threshold.setValue(preset["down_threshold"])
        self.up_threshold.setToolTip("Probability cutoff for calling a patch infected.")
        self.down_threshold.setToolTip("Probability cutoff for calling a patch clear.")
        self.spor_th.setToolTip("Sporulation-head cutoff for dual-head powdery mildew models.")

        self.cuda.setChecked(preset["cuda"])
        #self.cuda_id.setText(str(preset["cuda_id"]))
        self.pretrained.setChecked(preset["pretrained"])
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

    # ------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------
    def parse_space_or_comma_list(self, text):
        return text.replace(",", " ").split()

    def build_command(self):
        script = self.script_path.text().strip()
        means = self.parse_space_or_comma_list(self.means.text())
        stds = self.parse_space_or_comma_list(self.stds.text())

        selected_tray = self.trays.currentText().strip()
        trays = [selected_tray] if selected_tray else []

        selected_image_folder = self.img_folder.currentText().strip()

        cmd = [
            sys.executable,
            script,
            "--model_type", self.model_type.currentText(),
            "--model_path", self.model_path.text().strip(),
            "--dataset_path", self.dataset_path.text().strip(),
            "--loading_epoch", str(self.loading_epoch.value()),
            "--timestamp", self.timestamp.text().strip(),
            "--outdim", str(self.outdim.value()),
            "--up_threshold", str(self.up_threshold.value()),
            "--down_threshold", str(self.down_threshold.value()),
            "--dpi", str(self.dpi.value()),
            "--img_folder", selected_image_folder,
            "--group", self.group.text().strip(),
            #"--step_size", str(self.step_size.value()),
            "--sal_threshold", str(self.sal_threshold.value()),
            "--sal_thresh_method", self.sal_thresh_method.currentText(),
            "--sal_thresh_p", str(self.sal_thresh_p.value()),
            "--log", self.log_path.text().strip(),
            "--means", *means,
            "--stds", *stds,
        ]

        if trays:
            cmd += ["--trays", *trays]

        pm_value = self.pm.text().strip()
        if pm_value:
            cmd += ["--pm", pm_value]

        if self.cuda.isChecked():
            cmd += ["--cuda"]

        if self.pretrained.isChecked():
            cmd.append("--pretrained")

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

        return cmd

    def command_to_string(self, cmd):
        return " ".join(f'"{x}"' if " " in x else x for x in cmd)

    # ------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------
    def preview_command(self):
        cmd = self.build_command()
        self.output.clear()
        self.output.append("Command preview:\n")
        self.output.append(self.command_to_string(cmd))

    def copy_command(self):
        cmd = self.build_command()
        QApplication.clipboard().setText(self.command_to_string(cmd))
        self.output.append("\n\nCommand copied to clipboard.")

    def run_pipeline(self):
        cmd = self.build_command()

        self.output.clear()
        self.output.append("Running command:\n")
        self.output.append(self.command_to_string(cmd))
        self.output.append("\n\nOutput:\n")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if process.stdout is not None:
                for line in process.stdout:
                    self.output.append(line.rstrip())
                    QApplication.processEvents()

            process.wait()

            self.output.append(f"\nFinished with exit code {process.returncode}")

        except FileNotFoundError as e:
            self.output.append(f"\nERROR: Could not find file or executable:\n{e}")

        except Exception as e:
            self.output.append(f"\nERROR while running pipeline:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PowderyMildewGUI()
    window.resize(1000, 1000)
    window.show()
    sys.exit(app.exec())
