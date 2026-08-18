"""App-level settings: ComfyUI server URL, ComfyUI installation folder (used
to list local workflow files), and an optional override for where
comfy-prompt-tools lives (blank = auto-detect as a sibling directory)."""

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from funkytown_testing_harness.gui.app_settings import load_settings, save_settings


class _FolderPicker(QWidget):
    def __init__(self, initial_text="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.line_edit = QLineEdit(initial_text)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(browse_button)

    def _browse(self):
        start_dir = self.line_edit.text() or ""
        chosen = QFileDialog.getExistingDirectory(self, "Select folder", start_dir)
        if chosen:
            self.line_edit.setText(chosen)

    def text(self):
        return self.line_edit.text().strip()


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)
        self.settings = load_settings()

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.server_edit = QLineEdit(self.settings["server"])
        form.addRow("ComfyUI server URL:", self.server_edit)

        self.comfyui_dir_picker = _FolderPicker(self.settings["comfyui_install_dir"])
        form.addRow("ComfyUI installation folder:", self.comfyui_dir_picker)

        self.comfy_prompt_tools_dir_picker = _FolderPicker(self.settings["comfy_prompt_tools_dir"])
        form.addRow("comfy-prompt-tools folder\n(blank = auto-detect sibling):", self.comfy_prompt_tools_dir_picker)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        self.settings = {
            "server": self.server_edit.text().strip() or "http://127.0.0.1:8000",
            "comfyui_install_dir": self.comfyui_dir_picker.text(),
            "comfy_prompt_tools_dir": self.comfy_prompt_tools_dir_picker.text(),
        }
        save_settings(self.settings)
        self.accept()
