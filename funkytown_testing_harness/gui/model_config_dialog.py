"""Dialog for building one model's entry in the config: its filename plus a
list of KSampler override sets (one queued run per set - same as run_test.py's
"configs" list). Each field has its own checkbox: unchecked means "don't
override this, use the workflow's own value" - matching apply_ksampler_overrides
in run_test.py, which only touches the keys actually present in a config dict.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class KSamplerConfigRow:
    """One editable row of KSampler overrides, shown inside a QGroupBox."""

    FIELDS = ("sampler_name", "steps", "cfg", "scheduler", "seed", "denoise")

    def __init__(self, sampler_names, schedulers):
        self.box = QGroupBox()
        layout = QVBoxLayout(self.box)

        self.enabled_checks = {}
        self.widgets = {}

        self._add_combo_row(layout, "sampler_name", "Sampler", sampler_names)
        self._add_spin_row(layout, "steps", "Steps", minimum=1, maximum=200, default=20)
        self._add_double_row(layout, "cfg", "CFG", minimum=0.0, maximum=30.0, default=7.0)
        self._add_combo_row(layout, "scheduler", "Scheduler", schedulers)
        self._add_spin_row(layout, "seed", "Seed", minimum=0, maximum=2**31 - 1, default=0)
        self._add_double_row(layout, "denoise", "Denoise", minimum=0.0, maximum=1.0, default=1.0)

    def _add_combo_row(self, layout, key, label, options):
        row = QHBoxLayout()
        check = QCheckBox(label)
        combo = QComboBox()
        combo.addItems(options)
        combo.setEnabled(False)
        check.toggled.connect(combo.setEnabled)
        row.addWidget(check)
        row.addWidget(combo, 1)
        layout.addLayout(row)
        self.enabled_checks[key] = check
        self.widgets[key] = combo

    def _add_spin_row(self, layout, key, label, minimum, maximum, default):
        row = QHBoxLayout()
        check = QCheckBox(label)
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(default)
        spin.setEnabled(False)
        check.toggled.connect(spin.setEnabled)
        row.addWidget(check)
        row.addWidget(spin, 1)
        layout.addLayout(row)
        self.enabled_checks[key] = check
        self.widgets[key] = spin

    def _add_double_row(self, layout, key, label, minimum, maximum, default):
        row = QHBoxLayout()
        check = QCheckBox(label)
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(0.1)
        spin.setValue(default)
        spin.setEnabled(False)
        check.toggled.connect(spin.setEnabled)
        row.addWidget(check)
        row.addWidget(spin, 1)
        layout.addLayout(row)
        self.enabled_checks[key] = check
        self.widgets[key] = spin

    def to_overrides(self):
        overrides = {}
        for key in self.FIELDS:
            if self.enabled_checks[key].isChecked():
                widget = self.widgets[key]
                if isinstance(widget, QComboBox):
                    overrides[key] = widget.currentText()
                elif isinstance(widget, QSpinBox):
                    overrides[key] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    overrides[key] = widget.value()
        return overrides


class ModelConfigDialog(QDialog):
    """Configure one model's KSampler config(s). Result available via
    .model_name and .configs after exec() returns QDialog.Accepted."""

    def __init__(self, model_name, sampler_names, schedulers, parent=None, initial_configs=None):
        super().__init__(parent)
        self.setWindowTitle(f"Configure: {model_name}")
        self.setMinimumWidth(420)
        self.model_name = model_name
        self._sampler_names = sampler_names or ["euler"]
        self._schedulers = schedulers or ["normal"]
        self._rows = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{model_name}</b>"))
        layout.addWidget(QLabel(
            "Check a field to override it for this config; leave unchecked to use\n"
            "whatever the workflow's own KSampler already has. Add multiple configs\n"
            "to run this model more than once, each with different settings."
        ))

        self.configs_list_container = QVBoxLayout()
        layout.addLayout(self.configs_list_container)

        add_button = QPushButton("+ Add another config for this model")
        add_button.clicked.connect(lambda: self._add_row())
        layout.addWidget(add_button)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for config in (initial_configs or [{}]):
            self._add_row(config)

    def _add_row(self, initial=None):
        row = KSamplerConfigRow(self._sampler_names, self._schedulers)
        if initial:
            for key, value in initial.items():
                if key in row.enabled_checks:
                    row.enabled_checks[key].setChecked(True)
                    widget = row.widgets[key]
                    if isinstance(widget, QComboBox):
                        idx = widget.findText(str(value))
                        if idx >= 0:
                            widget.setCurrentIndex(idx)
                    else:
                        widget.setValue(value)

        remove_button = QPushButton("Remove this config")
        remove_button.clicked.connect(lambda: self._remove_row(row))
        row_layout = row.box.layout()
        row_layout.addWidget(remove_button)

        self._rows.append(row)
        self.configs_list_container.addWidget(row.box)

    def _remove_row(self, row):
        if len(self._rows) <= 1:
            QMessageBox.information(self, "Can't remove", "A model needs at least one config.")
            return
        self._rows.remove(row)
        row.box.setParent(None)
        row.box.deleteLater()

    def _on_accept(self):
        self.configs = [row.to_overrides() for row in self._rows]
        self.accept()
