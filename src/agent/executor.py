"""
Stateful code executors for local Python and QGIS environments.
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _norm(path: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expandvars(path)))


def _looks_like_qgis_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    return os.path.isdir(os.path.join(path, "python")) and os.path.isdir(os.path.join(path, "bin"))


def _find_qgis_app_dir(qgis_prefix: str) -> str:
    apps_dir = os.path.join(qgis_prefix, "apps")
    preferred = os.path.join(apps_dir, "qgis")
    if os.path.isdir(preferred):
        return preferred

    matches = [p for p in glob.glob(os.path.join(apps_dir, "qgis*")) if os.path.isdir(p)]
    if matches:
        return sorted(matches, key=lambda x: os.path.basename(x).lower())[0]
    return ""


def _resolve_qgis_prefix(path: str) -> str:
    if not path:
        return ""

    p = _norm(path)
    if not os.path.exists(p):
        return ""

    if os.path.isfile(p):
        name = os.path.basename(p).lower()
        parent = os.path.dirname(p)
        if name.startswith("python-qgis") and name.endswith(".bat") and os.path.basename(parent).lower() == "bin":
            return os.path.dirname(parent)
        if name == "python.exe" and os.path.basename(parent).lower().startswith("python"):
            apps_dir = os.path.dirname(parent)
            if os.path.basename(apps_dir).lower() == "apps":
                return os.path.dirname(apps_dir)
        return ""

    if _find_qgis_app_dir(p):
        return p

    if _looks_like_qgis_dir(p):
        parent = os.path.dirname(p)
        if os.path.basename(parent).lower() == "apps":
            return os.path.dirname(parent)
        return p

    for child in glob.glob(os.path.join(p, "QGIS*")):
        child_norm = _norm(child)
        if _find_qgis_app_dir(child_norm):
            return child_norm

    return ""


def _qgis_python_candidates(qgis_prefix: str) -> list[str]:
    if not qgis_prefix:
        return [sys.executable]

    candidates = sorted(
        glob.glob(os.path.join(qgis_prefix, "bin", "python-qgis*.bat")),
        reverse=True,
    )
    candidates.extend(
        sorted(
            glob.glob(os.path.join(qgis_prefix, "apps", "Python*", "python.exe")),
            reverse=True,
        )
    )
    candidates.extend(
        [
            os.path.join(qgis_prefix, "bin", "python3.exe"),
            os.path.join(qgis_prefix, "bin", "python.exe"),
            sys.executable,
        ]
    )

    uniq: list[str] = []
    seen = set()
    for c in candidates:
        n = _norm(c)
        if n in seen:
            continue
        seen.add(n)
        if c == sys.executable or os.path.exists(c):
            uniq.append(c)
    return uniq


def detect_qgis_prefix() -> str:
    """Detect likely QGIS installation prefix on Windows."""
    env_hint = os.environ.get("QGIS_PREFIX_PATH")
    if env_hint:
        resolved = _resolve_qgis_prefix(env_hint)
        if resolved:
            return resolved

    for pattern in [r"C:\Program Files\QGIS*", r"C:\OSGeo4W*", r"D:\QGIS*"]:
        for candidate in sorted(glob.glob(pattern), reverse=True):
            resolved = _resolve_qgis_prefix(candidate)
            if resolved:
                return resolved

    candidates = [
        r"C:\Program Files\QGIS 3.40.0",
        r"C:\Program Files\QGIS 3.38.0",
        r"C:\Program Files\QGIS 3.34.0",
        r"C:\Program Files\QGIS 3.28.0",
        r"C:\OSGeo4W",
        r"C:\OSGeo4W64",
        r"D:\QGIS",
    ]
    for candidate in candidates:
        resolved = _resolve_qgis_prefix(candidate)
        if resolved:
            return resolved
    return ""


def build_qgis_env(qgis_prefix: str, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build environment variables required for importing qgis.* modules."""
    env = dict(base_env or os.environ)
    if not qgis_prefix:
        return env

    resolved = _resolve_qgis_prefix(qgis_prefix)
    qgis_prefix = resolved or qgis_prefix

    apps_dir = os.path.join(qgis_prefix, "apps")
    qgis_dir = _find_qgis_app_dir(qgis_prefix) or qgis_prefix

    py_paths = [
        os.path.join(qgis_dir, "python"),
        os.path.join(qgis_dir, "python", "plugins"),
    ]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    bin_paths = [
        os.path.join(qgis_dir, "bin"),
        os.path.join(apps_dir, "Qt6", "bin"),
        os.path.join(apps_dir, "Qt5", "bin"),
        os.path.join(qgis_prefix, "bin"),
    ]
    if env.get("PATH"):
        bin_paths.append(env["PATH"])
    env["PATH"] = os.pathsep.join(bin_paths)
    env["QGIS_PREFIX_PATH"] = qgis_dir
    return env


def detect_qgis_python(qgis_prefix: str) -> str:
    """Detect Python executable suitable for QGIS worker process."""
    if not qgis_prefix:
        return sys.executable

    for candidate in _qgis_python_candidates(qgis_prefix):
        if candidate == sys.executable or os.path.exists(candidate):
            return candidate
    return sys.executable


class LocalKernel:
    """Stateful in-process Python kernel."""

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        python_path: Optional[str] = None,
        show_layer_callback: Optional[Callable[[str, Optional[str]], None]] = None,
    ):
        self.workspace_path = workspace_path or tempfile.gettempdir()
        os.makedirs(self.workspace_path, exist_ok=True)
        self.python_path = python_path or sys.executable
        self._show_layer_callback = show_layer_callback
        self.globals: Dict[str, Any] = {
            "__name__": "__main__",
            "__builtins__": __builtins__,
        }
        self._inject_helpers()

    def _inject_helpers(self) -> None:
        self.globals["show_layer"] = self._show_layer

    def set_show_layer_callback(self, callback: Optional[Callable[[str, Optional[str]], None]]) -> None:
        self._show_layer_callback = callback

    def reset(self) -> None:
        self.globals = {
            "__name__": "__main__",
            "__builtins__": __builtins__,
        }
        self._inject_helpers()

    def _show_layer(self, path: str, layer_name: Optional[str] = None) -> None:
        event = {"type": "show_layer", "path": path, "layer_name": layer_name}
        self._events.append(event)
        if self._show_layer_callback:
            self._show_layer_callback(path, layer_name)

    def execute(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        del kwargs
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        self._events = []
        cwd_before = os.getcwd()
        trace = ""

        try:
            os.chdir(self.workspace_path)
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, self.globals)
            status = "success"
        except Exception:
            status = "error"
            trace = traceback.format_exc()
            stderr_buf.write(trace)
        finally:
            os.chdir(cwd_before)

        return {
            "status": status,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "traceback": trace or None,
            "events": list(self._events),
            "executor": "LocalKernel",
        }


class LocalPythonExecutor(LocalKernel):
    """Backward-compatible alias for existing tool usage."""


class QGISPythonExecutor:
    """Stateful QGIS executor backed by a persistent worker process."""

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        qgis_prefix_path: Optional[str] = None,
    ):
        self.workspace_path = workspace_path or tempfile.gettempdir()
        os.makedirs(self.workspace_path, exist_ok=True)
        self.qgis_prefix = qgis_prefix_path or detect_qgis_prefix()
        self.env = build_qgis_env(self.qgis_prefix)
        self._worker: Optional[subprocess.Popen[str]] = None
        self._worker_path = str(Path(__file__).with_name("qgis_worker.py"))

    def get_python_executable(self) -> str:
        return detect_qgis_python(self.qgis_prefix)

    def _start_worker(self) -> None:
        if self._worker and self._worker.poll() is None:
            return

        if not self.qgis_prefix:
            raise RuntimeError("未找到 QGIS 安装路径")
        if not os.path.exists(self._worker_path):
            raise RuntimeError(f"QGIS worker 脚本不存在: {self._worker_path}")

        python_exe = self.get_python_executable()
        cmd = [python_exe, self._worker_path, "--stdio"]
        worker_env = None if python_exe.lower().endswith(".bat") else self.env
        self._worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.workspace_path,
            env=worker_env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        # Worker must emit exactly one ready line.
        ready_payload = {}
        ready_line = ""
        if self._worker and self._worker.stdout:
            while True:
                ready_line = self._worker.stdout.readline().strip()
                if not ready_line:
                    break
                try:
                    ready_payload = json.loads(ready_line)
                    break
                except json.JSONDecodeError:
                    # Ignore non-protocol log lines printed by QGIS runtime.
                    continue
        if ready_payload.get("type") != "ready":
            err_line = self._worker.stderr.readline().strip() if self._worker and self._worker.stderr else ""
            raise RuntimeError(f"QGIS worker 启动失败: {ready_line or err_line}")

    def execute(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        timeout = int(kwargs.get("timeout", 120))
        try:
            self._start_worker()
            if not self._worker or not self._worker.stdin or not self._worker.stdout:
                return {"status": "error", "message": "QGIS worker 不可用"}

            request_id = uuid.uuid4().hex
            payload = {"type": "execute", "id": request_id, "code": code, "timeout": timeout}
            self._worker.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._worker.stdin.flush()

            while True:
                line = self._worker.stdout.readline()
                if not line:
                    err = self._worker.stderr.read() if self._worker and self._worker.stderr else ""
                    return {"status": "error", "message": "QGIS worker 已退出", "stderr": err}
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == request_id:
                    msg.setdefault("executor", "QGISKernel")
                    return msg
        except Exception as exc:
            return {"status": "error", "message": str(exc), "executor": "QGISKernel"}

    def close(self) -> None:
        if not self._worker:
            return
        try:
            if self._worker.poll() is None and self._worker.stdin:
                self._worker.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                self._worker.stdin.flush()
        except Exception:
            pass
        finally:
            try:
                self._worker.terminate()
            except Exception:
                pass
            self._worker = None

    def __del__(self) -> None:
        self.close()
