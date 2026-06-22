from __future__ import annotations

from typing import Any, Dict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from src.ui.i18n import tr


class EnvStatusWidget(QWidget):
    """Compact status widget displayed in status bar."""

    config_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._python_status: Dict[str, Any] = {}
        self._qgis_status: Dict[str, Any] = {}
        self._build_ui()
        self.update_status({}, {})

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.python_btn = QPushButton()
        self.python_btn.setFlat(True)
        self.python_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.python_btn.clicked.connect(self.config_requested.emit)
        layout.addWidget(self.python_btn)

        self.qgis_btn = QPushButton()
        self.qgis_btn.setFlat(True)
        self.qgis_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.qgis_btn.clicked.connect(self.config_requested.emit)
        layout.addWidget(self.qgis_btn)

        self.config_btn = QPushButton(tr("common.config"))
        self.config_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.config_btn.clicked.connect(self.config_requested.emit)
        layout.addWidget(self.config_btn)

    def update_status(self, python_status: Dict[str, Any] | None, qgis_status: Dict[str, Any] | None) -> None:
        self._python_status = python_status or {}
        self._qgis_status = qgis_status or {}

        self.config_btn.setText(tr("common.config"))

        py_text, py_color = self._format_python(self._python_status)
        qgis_text, qgis_color = self._format_qgis(self._qgis_status)

        self.python_btn.setText(py_text)
        self.python_btn.setStyleSheet(f"color: {py_color};")
        self.python_btn.setToolTip(self._format_tooltip("Python", self._python_status))

        self.qgis_btn.setText(qgis_text)
        self.qgis_btn.setStyleSheet(f"color: {qgis_color};")
        self.qgis_btn.setToolTip(self._format_tooltip("QGIS", self._qgis_status))

    def retranslate_ui(self) -> None:
        self.update_status(self._python_status, self._qgis_status)

    def _format_python(self, status: Dict[str, Any]) -> tuple[str, str]:
        if not status:
            return f"Python: ? {tr('common.not_checked')}", "#a16207"
        if self._is_valid(status):
            version = status.get("version") or tr("common.available")
            return f"Python: OK {version}", "#15803d"
        msg = status.get("message") or tr("common.unavailable")
        return f"Python: ERROR {msg}", "#dc2626"

    def _format_qgis(self, status: Dict[str, Any]) -> tuple[str, str]:
        if not status:
            return f"QGIS: ? {tr('common.not_checked')}", "#a16207"
        if self._is_valid(status):
            version = status.get("qgis_version") or status.get("version") or tr("common.available")
            return f"QGIS: OK {version}", "#15803d"
        msg = status.get("message") or tr("common.unavailable")
        return f"QGIS: ERROR {msg}", "#dc2626"

    @staticmethod
    def _is_valid(status: Dict[str, Any]) -> bool:
        # Backward compatibility: old/new payloads may use valid or validated.
        if status.get("valid") is not None:
            return bool(status.get("valid"))
        return bool(status.get("validated"))

    def _format_tooltip(self, title: str, status: Dict[str, Any]) -> str:
        if not status:
            return f"{title}: {tr('common.not_checked')}"

        lines = [
            f"{title}",
            f"{tr('common.path')}: {status.get('path', 'N/A')}",
            f"{tr('common.message')}: {status.get('message', 'N/A')}",
        ]

        version = status.get("version") or status.get("qgis_version")
        if version:
            lines.append(f"{tr('common.version')}: {version}")

        last_check = status.get("last_check")
        if last_check:
            lines.append(f"{tr('common.last_check')}: {last_check}")

        libs = status.get("libs", {})
        if isinstance(libs, dict) and libs:
            lines.append(f"{tr('common.dependencies')}:")
            for lib, ok in libs.items():
                lines.append(f"  {'OK' if ok else 'NO'} {lib}")

        return "\n".join(lines)
