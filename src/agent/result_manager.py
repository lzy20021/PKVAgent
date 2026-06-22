import os
import re
from typing import Any, Callable, Dict, List, Optional


class ResultManager:
    """Parse executor results and expose user-friendly output payloads."""

    def __init__(self, ui_callbacks: Optional[Dict[str, Callable[..., Any]]] = None):
        self.ui = ui_callbacks or {}

    def handle_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if result.get("status") == "success":
            return self._handle_success(result)
        return self._handle_error(result)

    def _handle_success(self, result: Dict[str, Any]) -> Dict[str, Any]:
        output_files = self._extract_output_files(result)
        valid_files = [p for p in output_files if self._validate_file(p)]

        loaded = []
        for file_path in valid_files:
            if self._load_layer(file_path):
                loaded.append(file_path)

        stdout = (result.get("stdout") or "").strip()
        return {
            "status": "success",
            "message": f"Task completed, generated {len(valid_files)} file(s).",
            "output_files": valid_files,
            "loaded_files": loaded,
            "stdout": stdout,
        }

    def _handle_error(self, result: Dict[str, Any]) -> Dict[str, Any]:
        stderr = result.get("stderr") or result.get("message") or ""
        traceback_text = result.get("traceback") or stderr
        error_type = self._parse_error_type(traceback_text)
        suggestion = self._generate_suggestion(error_type, traceback_text)
        root_cause = ""
        if isinstance(traceback_text, str) and traceback_text.strip():
            root_cause = traceback_text.strip().splitlines()[-1]
        message_text = root_cause or stderr

        return {
            "status": "error",
            "message": f"Execution failed: {message_text[:320]}" if message_text else "Execution failed.",
            "error_type": error_type,
            "suggestion": suggestion,
            "traceback": traceback_text,
        }

    def _extract_output_files(self, result: Dict[str, Any]) -> List[str]:
        output_files: List[str] = []

        for event in result.get("events", []) or []:
            if isinstance(event, dict) and event.get("type") == "show_layer":
                path = event.get("path")
                if path:
                    output_files.append(path)

        stdout = result.get("stdout") or ""
        patterns = [
            r"saved\s*[:=]\s*([^\s]+\.(?:shp|geojson|gpkg))",
            r"output\s*[:=]\s*([^\s]+\.(?:shp|geojson|gpkg))",
            r"result\s*[:=]\s*([^\s]+\.(?:shp|geojson|gpkg))",
            r"([A-Za-z]:[\\/][^\r\n]+?\.(?:shp|geojson|gpkg))",
            r"([^\s\"']+\.(?:shp|geojson|gpkg))",
            r"saved\s*[:=]\s*([^\s]+\.(?:png|jpg|jpeg))",
            r"output\s*[:=]\s*([^\s]+\.(?:png|jpg|jpeg))",
            r"result\s*[:=]\s*([^\s]+\.(?:png|jpg|jpeg))",
            r"([A-Za-z]:[\\/][^\r\n]+?\.(?:png|jpg|jpeg))",
            r"([^\s\"']+\.(?:png|jpg|jpeg))",
            r"saved\s*[:=]\s*([^\s]+\.(?:pdf))",
            r"output\s*[:=]\s*([^\s]+\.(?:pdf))",
            r"result\s*[:=]\s*([^\s]+\.(?:pdf))",
            r"([A-Za-z]:[\\/][^\r\n]+?\.(?:pdf))",
            r"([^\s\"']+\.(?:pdf))",
        ]
        for pattern in patterns:
            output_files.extend(re.findall(pattern, stdout, flags=re.IGNORECASE))

        seen = set()
        deduped: List[str] = []
        for p in output_files:
            if p not in seen:
                deduped.append(p)
                seen.add(p)
        return deduped

    def _validate_file(self, file_path: str) -> bool:
        if not file_path:
            return False
        if not os.path.exists(file_path):
            return False
        if os.path.getsize(file_path) <= 0:
            return False

        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".pdf"}:
            return True

        # Lightweight validation; avoid hard dependency failure breaking pipeline.
        try:
            import geopandas as gpd  # type: ignore

            gdf = gpd.read_file(file_path)
            return len(gdf) > 0
        except Exception:
            # If GeoPandas cannot parse here, keep file as invalid to avoid loading bad outputs.
            return False

    def _load_layer(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in {".shp", ".geojson", ".gpkg"}:
            return False
        cb = self.ui.get("load_layer")
        if not callable(cb):
            return False

        try:
            layer_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                cb(file_path, layer_name)
            except TypeError:
                cb(file_path)
            return True
        except Exception:
            return False

    def _parse_error_type(self, traceback_text: str) -> str:
        text = (traceback_text or "").lower()
        mapping = {
            "filenotfounderror": "file_not_found",
            "modulenotfounderror": "module_not_found",
            "importerror": "module_not_found",
            "memoryerror": "memory_error",
            "valueerror": "value_error",
            "keyerror": "key_error",
            "attributeerror": "attribute_error",
            "crserror": "crs_error",
            "permissionerror": "permission_error",
            "timeout": "timeout_error",
        }
        for marker, kind in mapping.items():
            if marker in text:
                return kind
        return "unknown"

    def _generate_suggestion(self, error_type: str, traceback_text: str = "") -> str:
        suggestions = {
            "file_not_found": "Check input/output path and ensure the file exists in workspace.",
            "module_not_found": "Verify Python environment dependencies and reinstall missing packages.",
            "memory_error": "Reduce dataset size, process by chunks, or simplify geometry first.",
            "value_error": "Check parameter values and units (e.g. meter vs kilometer).",
            "key_error": "Check attribute field names and spelling.",
            "attribute_error": "Validate object type and geometry type compatibility.",
            "crs_error": "Check CRS definition and convert to projected CRS before metric operations.",
            "permission_error": "Use a writable output directory.",
            "timeout_error": "Simplify task or split into smaller steps to avoid timeout.",
            "unknown": "Inspect traceback details and retry with simpler parameters.",
        }

        suggestion = suggestions.get(error_type, suggestions["unknown"])
        if "No such file or directory" in traceback_text:
            match = re.search(r"'([^']+)'", traceback_text)
            if match:
                suggestion += f" Missing file: {match.group(1)}"
        return suggestion
