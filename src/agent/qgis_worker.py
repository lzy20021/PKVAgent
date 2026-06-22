"""
Persistent QGIS worker process.
Protocol: JSON lines over stdin/stdout.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import traceback
from typing import Any, Dict, List, Optional


def send(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def build_kernel_globals() -> Dict[str, Any]:
    namespace: Dict[str, Any] = {
        "__name__": "__main__",
        "__builtins__": __builtins__,
    }
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdio", action="store_true", help="Use stdio JSON protocol")
    _ = parser.parse_args()

    try:
        from qgis.core import QgsApplication
    except Exception:
        send(
            {
                "type": "ready",
                "status": "error",
                "message": "无法导入 qgis.core",
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    qgs = QgsApplication([], False)
    qgs.initQgis()

    namespace = build_kernel_globals()
    send({"type": "ready", "status": "ok"})

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                send({"type": "error", "status": "error", "message": "Invalid JSON"})
                continue

            if req.get("type") == "shutdown":
                break
            if req.get("type") != "execute":
                send({"type": "error", "status": "error", "message": "Unknown command"})
                continue

            request_id = req.get("id")
            code = req.get("code", "")
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            events: List[Dict[str, Any]] = []

            def show_layer(path: str, layer_name: Optional[str] = None) -> None:
                events.append({"type": "show_layer", "path": path, "layer_name": layer_name})

            namespace["show_layer"] = show_layer
            trace = ""
            status = "success"
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                try:
                    exec(code, namespace)
                except Exception:
                    status = "error"
                    trace = traceback.format_exc()
                    stderr_buf.write(trace)

            send(
                {
                    "type": "result",
                    "id": request_id,
                    "status": status,
                    "stdout": stdout_buf.getvalue(),
                    "stderr": stderr_buf.getvalue(),
                    "traceback": trace or None,
                    "events": events,
                }
            )
    finally:
        qgs.exitQgis()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
