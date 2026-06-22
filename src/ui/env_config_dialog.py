from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

import yaml
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from src.ui.i18n import tr
from src.utils.env_check import auto_detect_python, auto_detect_qgis, check_python_env, check_qgis_env


class EnvConfigDialog(QDialog):
    """Environment configuration dialog."""

    env_config_changed = Signal(dict)

    def __init__(self, parent=None, config: Dict[str, Any] | None = None, config_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr("env.title"))
        self.setMinimumWidth(700)

        self.config_path = config_path or self._default_config_path()
        self.config: Dict[str, Any] = config.copy() if isinstance(config, dict) else self._load_config()
        self._python_result: Dict[str, Any] = {"valid": False, "message": tr("env.unchecked")}
        self._qgis_result: Dict[str, Any] = {"valid": False, "message": tr("env.unchecked")}

        self._build_ui()
        self._load_from_config()

    def _default_config_path(self) -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base, "config.yaml")

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        root.addWidget(self._create_python_group())
        root.addWidget(self._create_qgis_group())

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setVisible(False)
        self.log_text.setMaximumHeight(180)
        root.addWidget(self.log_text)

        actions = QHBoxLayout()

        self.log_btn = QPushButton(tr("env.view_log"))
        self.log_btn.clicked.connect(self.toggle_log)
        actions.addWidget(self.log_btn)

        actions.addStretch()

        verify_btn = QPushButton(tr("env.verify_now"))
        verify_btn.clicked.connect(self.verify)
        actions.addWidget(verify_btn)

        save_btn = QPushButton(tr("common.save"))
        save_btn.clicked.connect(self.save_config)
        actions.addWidget(save_btn)

        cancel_btn = QPushButton(tr("common.cancel"))
        cancel_btn.clicked.connect(self.reject)
        actions.addWidget(cancel_btn)

        root.addLayout(actions)

    def _create_python_group(self) -> QGroupBox:
        group = QGroupBox(tr("env.python_group"))
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel(f"{tr('common.path')}:"))

        self.python_edit = QLineEdit()
        self.python_edit.setPlaceholderText(tr("env.python_placeholder"))
        self.python_edit.textChanged.connect(self.validate_python)
        row.addWidget(self.python_edit)

        browse_btn = QPushButton(tr("common.browse"))
        browse_btn.clicked.connect(self.browse_python)
        row.addWidget(browse_btn)

        detect_btn = QPushButton(tr("common.detect"))
        detect_btn.clicked.connect(self.detect_python)
        row.addWidget(detect_btn)

        layout.addLayout(row)

        self.python_status_label = QLabel(f"{tr('env.status')}: ? {tr('env.unchecked')}")
        layout.addWidget(self.python_status_label)

        return group

    def _create_qgis_group(self) -> QGroupBox:
        group = QGroupBox(tr("env.qgis_group"))
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel(f"{tr('common.path')}:"))

        self.qgis_edit = QLineEdit()
        self.qgis_edit.setPlaceholderText(tr("env.qgis_placeholder"))
        self.qgis_edit.textChanged.connect(self.validate_qgis)
        row.addWidget(self.qgis_edit)

        browse_btn = QPushButton(tr("common.browse"))
        browse_btn.clicked.connect(self.browse_qgis)
        row.addWidget(browse_btn)

        detect_btn = QPushButton(tr("common.detect"))
        detect_btn.clicked.connect(self.detect_qgis)
        row.addWidget(detect_btn)

        layout.addLayout(row)

        self.qgis_status_label = QLabel(f"{tr('env.status')}: ? {tr('env.unchecked')}")
        layout.addWidget(self.qgis_status_label)

        return group

    def _load_from_config(self) -> None:
        python_path = self.config.get("python_path") or self.config.get("python_env", {}).get("path") or ""
        qgis_path = self.config.get("qgis_path") or self.config.get("qgis_env", {}).get("path") or ""

        if not python_path:
            python_path = auto_detect_python() or ""
        if not qgis_path:
            qgis_path = auto_detect_qgis() or ""

        self.python_edit.setText(python_path)
        self.qgis_edit.setText(qgis_path)

        self.verify()

    def browse_python(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("env.python_file_title"), "", tr("env.python_file_filter"))
        if path:
            self.python_edit.setText(path)

    def browse_qgis(self) -> None:
        path = QFileDialog.getExistingDirectory(self, tr("env.qgis_dir_title"))
        if path:
            self.qgis_edit.setText(path)

    def detect_python(self) -> None:
        path = auto_detect_python() or ""
        self.python_edit.setText(path)
        self._append_log(tr("env.auto_detect_python", path=path or tr("env.not_found")))

    def detect_qgis(self) -> None:
        path = auto_detect_qgis() or ""
        self.qgis_edit.setText(path)
        self._append_log(tr("env.auto_detect_qgis", path=path or tr("env.not_found")))

    def verify(self) -> None:
        self.validate_python()
        self.validate_qgis()

    def validate_python(self) -> None:
        path = self.python_edit.text().strip()
        if not path:
            self._python_result = {"valid": False, "message": tr("env.path_empty"), "path": ""}
        else:
            try:
                self._python_result = check_python_env(path)
            except Exception as exc:
                self._python_result = {"valid": False, "message": str(exc), "path": path}

        self._python_result["path"] = path
        self._render_python_status(self._python_result)
        self._append_log(tr("env.check_python", result=self._python_result))

    def validate_qgis(self) -> None:
        path = self.qgis_edit.text().strip()
        if not path:
            self._qgis_result = {"valid": False, "message": tr("env.optional_unconfigured"), "path": ""}
        else:
            try:
                self._qgis_result = check_qgis_env(path)
            except Exception as exc:
                self._qgis_result = {"valid": False, "message": str(exc), "path": path}

        self._qgis_result["path"] = path
        self._render_qgis_status(self._qgis_result)
        self._append_log(tr("env.check_qgis", result=self._qgis_result))

    def _render_python_status(self, result: Dict[str, Any]) -> None:
        if result.get("valid"):
            version = result.get("version", "")
            libs = result.get("libs", {})
            ok_libs = [name for name, ok in libs.items() if ok]
            libs_text = ", ".join(ok_libs) if ok_libs else tr("env.no_optional_libs")
            self.python_status_label.setText(f"{tr('env.status')}: OK {version} | {tr('env.libs')}: {libs_text}")
            self.python_status_label.setStyleSheet("color: #15803d;")
        else:
            message = result.get("message", tr("common.unavailable"))
            self.python_status_label.setText(f"{tr('env.status')}: ERROR {message}")
            self.python_status_label.setStyleSheet("color: #dc2626;")

    def _render_qgis_status(self, result: Dict[str, Any]) -> None:
        if result.get("valid"):
            py_exe = result.get("python_exe", "")
            version = result.get("qgis_version", "")
            text = f"{tr('env.status')}: OK {'QGIS ' + version if version else 'QGIS ' + tr('common.available')}"
            if py_exe:
                text += f" | Python: {py_exe}"
            self.qgis_status_label.setText(text)
            self.qgis_status_label.setStyleSheet("color: #15803d;")
        else:
            message = result.get("message", tr("common.unavailable"))
            self.qgis_status_label.setText(f"{tr('env.status')}: ERROR {message}")
            self.qgis_status_label.setStyleSheet("color: #dc2626;")

    def toggle_log(self) -> None:
        shown = not self.log_text.isVisible()
        self.log_text.setVisible(shown)
        self.log_btn.setText(tr("env.hide_log") if shown else tr("env.view_log"))

    def _append_log(self, text: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{now}] {text}")

    def _build_saved_config(self) -> Dict[str, Any]:
        self.verify()
        now = datetime.now().isoformat(timespec="seconds")

        merged = dict(self.config)

        python_path = self.python_edit.text().strip()
        qgis_path = self.qgis_edit.text().strip()

        merged["python_path"] = python_path
        merged["qgis_path"] = qgis_path

        merged["python_env"] = {
            "path": python_path,
            "valid": bool(self._python_result.get("valid")),
            "validated": bool(self._python_result.get("valid")),
            "last_check": now,
            "version": self._python_result.get("version", ""),
            "libs": self._python_result.get("libs", {}),
            "message": self._python_result.get("message", ""),
        }

        merged["qgis_env"] = {
            "path": qgis_path,
            "valid": bool(self._qgis_result.get("valid")),
            "validated": bool(self._qgis_result.get("valid")),
            "last_check": now,
            "python_exe": self._qgis_result.get("python_exe", ""),
            "qgis_version": self._qgis_result.get("qgis_version", ""),
            "message": self._qgis_result.get("message", ""),
        }

        return merged

    def save_config(self) -> None:
        merged = self._build_saved_config()

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)

        self.config = merged
        self.env_config_changed.emit(merged)
        self.accept()

    def get_values(self) -> Dict[str, str]:
        """Backward-compatible helper for existing callers."""
        return {
            "python_path": self.python_edit.text().strip(),
            "qgis_path": self.qgis_edit.text().strip(),
        }
