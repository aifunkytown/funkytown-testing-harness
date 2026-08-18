"""Main window: pick a workflow, optionally strip LoRAs / override the
prompt, build up a list of models (each with its own KSampler config(s)), and
run - reusing run_test.run() and live_workflow.py exactly as the CLI does, so
behavior (model presence checking, the <2-models error, always-fresh fetch)
is identical either way.
"""

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from funkytown_testing_harness.gui import comfy_client
from funkytown_testing_harness.gui.app_settings import load_settings
from funkytown_testing_harness.gui.model_config_dialog import ModelConfigDialog
from funkytown_testing_harness.gui.runner_thread import TestRunnerThread
from funkytown_testing_harness.gui.settings_dialog import SettingsDialog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GUI_RUN_CONFIG_PATH = PROJECT_ROOT / "gui_last_run.json"
CONFIGS_DIR = PROJECT_ROOT / "configs"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("funkytown-testing-harness")
        self.resize(760, 720)

        self.settings = load_settings()
        self._models = {}  # model_name -> list[dict] (configs)
        self._runner_thread = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addLayout(self._build_top_bar())
        root.addWidget(self._build_workflow_group())
        root.addWidget(self._build_models_group(), 1)

        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, 1)

        self.run_button = QPushButton("Run Test")
        self.run_button.clicked.connect(self._on_run_clicked)
        root.addWidget(self.run_button)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        root.addWidget(QLabel("Log:"))
        root.addWidget(self.log_view, 1)

        self._refresh_workflow_list()
        self._refresh_model_dropdown()
        self._refresh_sampler_options()

    # ---- top bar (name, server status, settings) -------------------------

    def _build_top_bar(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Test name:"))
        self.name_edit = QLineEdit("model_testing")
        row.addWidget(self.name_edit, 1)

        self.server_label = QLabel()
        row.addWidget(self.server_label)

        settings_button = QPushButton("Settings...")
        settings_button.clicked.connect(self._open_settings)
        row.addWidget(settings_button)

        load_button = QPushButton("Load Config...")
        load_button.clicked.connect(self._load_config)
        row.addWidget(load_button)

        save_button = QPushButton("Save Config...")
        save_button.clicked.connect(self._save_config)
        row.addWidget(save_button)

        self._update_server_label()
        return row

    def _update_server_label(self):
        reachable = comfy_client.check_server_reachable(self.settings["server"])
        status = "reachable" if reachable else "NOT reachable"
        self.server_label.setText(f"Server: {self.settings['server']} ({status})")

    def _open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            self.settings = dialog.settings
            self._update_server_label()
            self._refresh_workflow_list()
            self._refresh_model_dropdown()
            self._refresh_sampler_options()

    # ---- workflow group ----------------------------------------------------

    def _build_workflow_group(self):
        group = QGroupBox("Workflow")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Source workflow:"))
        self.workflow_combo = QComboBox()
        self.workflow_combo.setEditable(True)
        row.addWidget(self.workflow_combo, 1)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_workflow_list)
        row.addWidget(refresh_button)
        layout.addLayout(row)

        self.strip_loras_check = QCheckBox("Strip LoRAs (clear the Power Lora Loader node)")
        layout.addWidget(self.strip_loras_check)

        layout.addWidget(QLabel("Positive prompt override (leave blank to use the workflow's own):"))
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setMaximumHeight(90)
        layout.addWidget(self.prompt_edit)

        return group

    def _refresh_workflow_list(self):
        current = self.workflow_combo.currentText()
        self.workflow_combo.clear()
        names = comfy_client.list_local_workflows(self.settings["comfyui_install_dir"])
        if not names:
            self.workflow_combo.addItem("")
            self._log("No workflows found - set the ComfyUI installation folder in Settings.")
        else:
            self.workflow_combo.addItems(names)
        if current:
            idx = self.workflow_combo.findText(current)
            if idx >= 0:
                self.workflow_combo.setCurrentIndex(idx)
            else:
                self.workflow_combo.setEditText(current)

    # ---- models group -------------------------------------------------------

    def _build_models_group(self):
        group = QGroupBox("Models to compare (at least 2 required)")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        row.addWidget(self.model_combo, 1)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_model_dropdown)
        row.addWidget(refresh_button)
        add_button = QPushButton("Add to list")
        add_button.clicked.connect(self._on_add_model)
        row.addWidget(add_button)
        layout.addLayout(row)

        self.models_list = QListWidget()
        self.models_list.itemDoubleClicked.connect(lambda item: self._edit_model(item.text()))
        layout.addWidget(self.models_list)

        buttons_row = QHBoxLayout()
        edit_button = QPushButton("Edit selected...")
        edit_button.clicked.connect(self._edit_selected)
        buttons_row.addWidget(edit_button)
        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self._remove_selected)
        buttons_row.addWidget(remove_button)
        layout.addLayout(buttons_row)

        return group

    def _refresh_model_dropdown(self):
        self.model_combo.clear()
        models = comfy_client.list_available_models(self.settings["server"])
        if not models:
            self._log("No models found - is ComfyUI running and reachable?")
        else:
            self.model_combo.addItems(models)

    def _refresh_sampler_options(self):
        self._sampler_names = comfy_client.list_sampler_names(self.settings["server"]) or ["euler"]
        self._schedulers = comfy_client.list_schedulers(self.settings["server"]) or ["normal"]

    def _on_add_model(self):
        model_name = self.model_combo.currentText().strip()
        if not model_name:
            return
        initial = self._models.get(model_name, [{}])
        dialog = ModelConfigDialog(model_name, self._sampler_names, self._schedulers, self, initial_configs=initial)
        if dialog.exec():
            self._models[model_name] = dialog.configs
            self._rebuild_models_list()

    def _edit_selected(self):
        item = self.models_list.currentItem()
        if item:
            self._edit_model(item.text())

    def _edit_model(self, list_text):
        model_name = list_text.split("  —")[0].strip()
        if model_name not in self._models:
            return
        dialog = ModelConfigDialog(
            model_name, self._sampler_names, self._schedulers, self, initial_configs=self._models[model_name]
        )
        if dialog.exec():
            self._models[model_name] = dialog.configs
            self._rebuild_models_list()

    def _remove_selected(self):
        item = self.models_list.currentItem()
        if not item:
            return
        model_name = item.text().split("  —")[0].strip()
        self._models.pop(model_name, None)
        self._rebuild_models_list()

    def _rebuild_models_list(self):
        self.models_list.clear()
        for model_name, configs in self._models.items():
            n = len(configs)
            label = f"{model_name}  — {n} config{'s' if n != 1 else ''}"
            self.models_list.addItem(QListWidgetItem(label))

    # ---- config assembly / save / load --------------------------------------

    def _build_config_dict(self):
        return {
            "name": self.name_edit.text().strip() or "model_testing",
            "source_workflow": self.workflow_combo.currentText().strip(),
            "strip_loras": self.strip_loras_check.isChecked(),
            "positive_prompt": self.prompt_edit.toPlainText().strip(),
            "server": self.settings["server"],
            "models": [
                {"model": model_name, "configs": configs}
                for model_name, configs in self._models.items()
            ],
        }

    def _save_config(self):
        CONFIGS_DIR.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", str(CONFIGS_DIR / "model-testing-config.json"), "JSON files (*.json)"
        )
        if not path:
            return
        config = self._build_config_dict()
        if not config["positive_prompt"]:
            del config["positive_prompt"]
        Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")
        self._log(f"Saved config to {path}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", str(CONFIGS_DIR), "JSON files (*.json)")
        if not path:
            return
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        self.name_edit.setText(config.get("name", "model_testing"))
        self.workflow_combo.setEditText(config.get("source_workflow", ""))
        self.strip_loras_check.setChecked(bool(config.get("strip_loras")))
        self.prompt_edit.setPlainText(config.get("positive_prompt", ""))
        self._models = {m["model"]: m.get("configs") or [{}] for m in config.get("models", [])}
        self._rebuild_models_list()
        self._log(f"Loaded config from {path}")

    # ---- run -----------------------------------------------------------------

    def _on_run_clicked(self):
        if not self.workflow_combo.currentText().strip():
            QMessageBox.warning(self, "No workflow selected", "Pick a source workflow first.")
            return
        if len(self._models) < 2:
            QMessageBox.warning(self, "Not enough models", "Add at least 2 models to compare.")
            return

        config = self._build_config_dict()
        if not config["positive_prompt"]:
            del config["positive_prompt"]
        GUI_RUN_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

        self.log_view.clear()
        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")

        self._runner_thread = TestRunnerThread(GUI_RUN_CONFIG_PATH, self)
        self._runner_thread.log_line.connect(self._log)
        self._runner_thread.finished_ok.connect(self._on_run_finished_ok)
        self._runner_thread.finished_error.connect(self._on_run_finished_error)
        self._runner_thread.start()

    def _on_run_finished_ok(self):
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Test")
        self._log("Done.")

    def _on_run_finished_error(self, message):
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Test")
        self._log(f"ERROR: {message}")
        QMessageBox.critical(self, "Run failed", message)

    def _log(self, line):
        self.log_view.appendPlainText(line)
