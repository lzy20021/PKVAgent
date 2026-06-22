"""MapGenAI application entry."""
import os
import sys
from datetime import datetime

import yaml
from PySide6.QtCore import QCoreApplication, Qt

from src.ui.main_window import main
from src.utils.env_check import auto_detect_python, auto_detect_qgis, check_python_env, check_qgis_env

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# QtWebEngine software rendering fallback.
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
os.environ["QT_QUICK_BACKEND"] = "software"
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, True)


def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save_config(config_path: str, config: dict) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def _validate_runtime_env(config_path: str) -> None:
    config = _load_config(config_path)
    now = datetime.now().isoformat(timespec="seconds")

    python_path = config.get("python_path") or config.get("python_env", {}).get("path") or auto_detect_python()
    qgis_path = config.get("qgis_path") or config.get("qgis_env", {}).get("path") or auto_detect_qgis()

    py_result = check_python_env(python_path)
    qgis_result = check_qgis_env(qgis_path) if qgis_path else {"valid": False}

    config["python_path"] = python_path
    config["qgis_path"] = qgis_path
    config["python_env"] = {
        "path": python_path,
        "validated": bool(py_result.get("valid")),
        "last_check": now,
        "version": py_result.get("version", ""),
        "libs": py_result.get("libs", {}),
        "message": py_result.get("message", ""),
    }
    config["qgis_env"] = {
        "path": qgis_path,
        "validated": bool(qgis_result.get("valid")),
        "last_check": now,
        "python_exe": qgis_result.get("python_exe", ""),
        "qgis_version": qgis_result.get("qgis_version", ""),
        "message": qgis_result.get("message", ""),
    }
    _save_config(config_path, config)


if __name__ == "__main__":
    _validate_runtime_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    main()
