import glob
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional


def _run_command(cmd: List[str], env: Optional[Dict[str, str]] = None, timeout: int = 15) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(exc)}


def _python_probe(python_path: str, modules: List[str]) -> Dict[str, Any]:
    probe = (
        "import importlib, json\n"
        f"mods = {modules!r}\n"
        "missing=[]\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception:\n"
        "        missing.append(m)\n"
        "print(json.dumps({'missing': missing}, ensure_ascii=False))\n"
    )
    result = _run_command([python_path, "-c", probe], timeout=20)
    if not result["ok"]:
        return {"ok": False, "missing": modules, "message": result["stderr"] or result["stdout"]}

    try:
        import json

        payload = json.loads(result["stdout"] or "{}")
        missing = payload.get("missing", [])
    except Exception:
        missing = modules
    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "message": "OK" if len(missing) == 0 else f"Missing: {', '.join(missing)}",
    }


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

    # 用户误选了可执行文件时，尝试反推安装根目录。
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

    # 用户选择了安装根目录，例如 C:\Program Files\QGIS 3.40.0
    if _find_qgis_app_dir(p):
        return p

    # 用户选择了 apps\qgis 目录
    if _looks_like_qgis_dir(p):
        parent = os.path.dirname(p)
        if os.path.basename(parent).lower() == "apps":
            return os.path.dirname(parent)
        return p

    # 用户选择了父目录（例如 D:\QGIS），目录下包含多个 QGIS 版本
    for child in glob.glob(os.path.join(p, "QGIS*")):
        child_norm = _norm(child)
        if _find_qgis_app_dir(child_norm):
            return child_norm

    return ""


def _qgis_python_candidates(qgis_prefix: str) -> List[str]:
    if not qgis_prefix:
        return [sys.executable]

    candidates: List[str] = []
    candidates.extend(
        sorted(
            glob.glob(os.path.join(qgis_prefix, "bin", "python-qgis*.bat")),
            reverse=True,
        )
    )
    candidates.extend(
        sorted(
            glob.glob(os.path.join(qgis_prefix, "apps", "Python*", "python.exe")),
            reverse=True,
        )
    )
    candidates.append(os.path.join(qgis_prefix, "bin", "python3.exe"))
    candidates.append(os.path.join(qgis_prefix, "bin", "python.exe"))
    candidates.append(sys.executable)

    uniq: List[str] = []
    seen = set()
    for path in candidates:
        n = _norm(path)
        if n in seen:
            continue
        seen.add(n)
        if path == sys.executable or os.path.exists(path):
            uniq.append(path)
    return uniq


def check_python_env(python_path: str) -> Dict[str, Any]:
    if not python_path:
        return {"valid": False, "message": "Python 路径为空"}
    if not os.path.exists(python_path):
        return {"valid": False, "message": "Python 路径不存在"}

    ver = _run_command([python_path, "--version"], timeout=5)
    if not ver["ok"] and not ver["stdout"]:
        return {"valid": False, "message": ver["stderr"] or "无法执行 Python"}

    modules = ["geopandas", "shapely", "fiona", "pandas", "pyproj"]
    probe = _python_probe(python_path, modules)
    return {
        "valid": probe["ok"],
        "version": ver["stdout"] or ver["stderr"],
        "libs": {m: m not in probe["missing"] for m in modules},
        "message": "环境完整" if probe["ok"] else probe["message"],
    }


def _build_qgis_env(qgis_prefix: str) -> Dict[str, str]:
    env = os.environ.copy()
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

    paths = [
        os.path.join(qgis_dir, "bin"),
        os.path.join(apps_dir, "Qt6", "bin"),
        os.path.join(apps_dir, "Qt5", "bin"),
        os.path.join(qgis_prefix, "bin"),
    ]
    if env.get("PATH"):
        paths.append(env["PATH"])
    env["PATH"] = os.pathsep.join(paths)
    env["QGIS_PREFIX_PATH"] = qgis_dir
    return env


def _detect_qgis_python(qgis_prefix: str) -> str:
    for path in _qgis_python_candidates(qgis_prefix):
        if path == sys.executable or os.path.exists(path):
            return path
    return sys.executable


def check_qgis_env(qgis_prefix: str) -> Dict[str, Any]:
    if not qgis_prefix:
        return {"valid": False, "message": "QGIS 路径为空"}

    resolved_prefix = _resolve_qgis_prefix(qgis_prefix)
    if not resolved_prefix:
        return {
            "valid": False,
            "message": "未识别为有效 QGIS 安装目录，请选择包含 apps/qgis 或 apps/qgis-ltr 的安装根目录",
        }

    env = _build_qgis_env(resolved_prefix)
    probe_code = (
        "import json\n"
        "try:\n"
        "    from qgis.core import QgsApplication, Qgis\n"
        "    version = getattr(Qgis, 'QGIS_VERSION', '')\n"
        "    if not version:\n"
        "        try:\n"
        "            version = QgsApplication.qgisVersion()\n"
        "        except Exception:\n"
        "            version = ''\n"
        "    print(json.dumps({'ok': True, 'message': 'qgis.core import success', 'qgis_version': version}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'message': str(e)}))\n"
        "    raise SystemExit(1)\n"
    )

    last_error = "QGIS 探针失败"
    for python_exe in _qgis_python_candidates(resolved_prefix):
        run_env = None if python_exe.lower().endswith(".bat") else env
        script_path = ""
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(probe_code)
                script_path = f.name
            res = _run_command([python_exe, script_path], env=run_env, timeout=20)
        finally:
            if script_path and os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except OSError:
                    pass
        if not res["ok"]:
            last_error = res["stderr"] or res["stdout"] or "QGIS 探针失败"
            continue

        try:
            import json

            payload = json.loads(res["stdout"] or "{}")
        except Exception:
            payload = {"ok": False, "message": res["stdout"] or "QGIS 探针解析失败"}

        if payload.get("ok"):
            return {
                "valid": True,
                "path": resolved_prefix,
                "python_exe": python_exe,
                "qgis_version": payload.get("qgis_version", ""),
                "message": payload.get("message", "QGIS 检查完成"),
            }

        last_error = payload.get("message") or last_error

    return {
        "valid": False,
        "path": resolved_prefix,
        "python_exe": _detect_qgis_python(resolved_prefix),
        "qgis_version": "",
        "message": last_error,
    }


def auto_detect_python() -> str:
    candidates = []
    if shutil.which("python"):
        candidates.append(shutil.which("python"))
    if shutil.which("python3"):
        candidates.append(shutil.which("python3"))
    candidates.extend(
        [
            sys.executable,
            r"C:\ProgramData\Anaconda3\python.exe",
            r"C:\ProgramData\Miniconda3\python.exe",
            rf"C:\Users\{os.environ.get('USERNAME', 'Admin')}\anaconda3\python.exe",
            r"D:\ProgramData\anaconda3\python.exe",
        ]
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return sys.executable


def auto_detect_qgis() -> str:
    env_hint = os.environ.get("QGIS_PREFIX_PATH")
    if env_hint:
        resolved = _resolve_qgis_prefix(env_hint)
        if resolved:
            return resolved

    search_patterns = [
        r"C:\Program Files\QGIS*",
        r"C:\OSGeo4W*",
        r"D:\QGIS*",
    ]
    for pattern in search_patterns:
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
