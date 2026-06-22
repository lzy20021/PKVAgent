import ast
from typing import Any, Dict, List


class QualityChecker:
    DANGEROUS_FUNCTIONS = {"eval", "exec", "compile", "__import__"}
    DANGEROUS_ATTR_CALLS = {
        "os.system",
        "subprocess.call",
        "subprocess.run",
        "subprocess.Popen",
        "os.popen",
    }

    def check(self, code: str) -> Dict[str, Any]:
        issues: List[str] = []
        warnings: List[str] = []

        tree = self._syntax_check(code)
        if tree is None:
            return {"valid": False, "issues": ["语法错误: 代码无法解析"], "warnings": []}

        issues.extend(self._check_security(tree))
        warnings.extend(self._check_logic(tree, code))
        warnings.extend(self._check_dependencies(tree))

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    def _syntax_check(self, code: str):
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    def _check_security(self, tree: ast.AST) -> List[str]:
        issues: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node)
                if name in self.DANGEROUS_FUNCTIONS:
                    issues.append(f"禁止使用危险函数: {name}")
                if name in self.DANGEROUS_ATTR_CALLS:
                    issues.append(f"禁止使用危险函数: {name}")

        return issues

    def _check_logic(self, tree: ast.AST, code: str) -> List[str]:
        warnings: List[str] = []

        has_output = False
        has_show_layer = "show_layer(" in code
        has_try_except = False
        has_crs_check = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node)
                if name in {"print", "show_layer"}:
                    has_output = True
            if isinstance(node, ast.Try):
                has_try_except = True

        code_text = code.lower()
        if "crs" in code_text or "to_crs" in code_text or "epsg" in code_text:
            has_crs_check = True

        if not has_output:
            warnings.append("警告: 代码没有输出结果")
        if not has_show_layer:
            warnings.append("警告: 未调用 show_layer 输出图层")
        if not has_try_except:
            warnings.append("警告: 缺少 try/except 错误处理")
        if "gpd.read_file" in code and not has_crs_check:
            warnings.append("警告: 缺少坐标系检查或转换")

        return warnings

    def _check_dependencies(self, tree: ast.AST) -> List[str]:
        warnings: List[str] = []
        imported: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

        expected = {"geopandas", "shapely", "qgis", "os", "json", "re"}
        has_gis_dep = any(mod.split(".")[0] in {"geopandas", "qgis", "shapely"} for mod in imported)
        if not has_gis_dep:
            warnings.append("警告: 未检测到GIS相关依赖导入（geopandas/qgis/shapely）")

        unknown = [m for m in imported if m.split(".")[0] not in expected and not m.startswith("src")]
        if unknown:
            warnings.append(f"警告: 检测到非常见依赖导入: {', '.join(sorted(set(unknown)))}")

        return warnings

    def _call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts = []
            cur = node.func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
        return ""
