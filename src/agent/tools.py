"""
地图综合工具定义
Agent可调用的所有工具
"""
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
import os
from .executor import LocalPythonExecutor, QGISPythonExecutor
from .skills import skill_manager

# 全局执行器实例
local_executor = LocalPythonExecutor()
qgis_executor = QGISPythonExecutor()
todo_state: List[Dict[str, Any]] = []

# UI回调注册表 - 用于与主界面交互
ui_callbacks: Dict[str, Callable] = {}

def register_ui_callback(name: str, func: Callable):
    """注册UI回调函数"""
    ui_callbacks[name] = func


def _get_path_from_config(config: Dict[str, Any], env_key: str, legacy_key: str) -> str:
    env_cfg = config.get(env_key, {})
    if isinstance(env_cfg, dict):
        value = env_cfg.get("path", "")
        if value:
            return value
    return config.get(legacy_key, "")


def _dispatch_layer_events(result: Dict[str, Any]) -> None:
    if not isinstance(result, dict):
        return
    events = result.get("events", [])
    if not isinstance(events, list):
        return
    if "load_layer" not in ui_callbacks:
        return
    for event in events:
        if isinstance(event, dict) and event.get("type") == "show_layer":
            file_path = event.get("path")
            layer_name = event.get("layer_name")
            if file_path:
                ui_callbacks["load_layer"](file_path, layer_name)


local_executor.set_show_layer_callback(
    lambda path, layer_name=None: ui_callbacks.get("load_layer", lambda *_: None)(path, layer_name)
)

def configure_tools(config: Dict):
    """根据配置初始化工具"""
    # 配置QGIS路径
    qgis_path = _get_path_from_config(config, "qgis_env", "qgis_path")
    if qgis_path:
        global qgis_executor
        qgis_executor = QGISPythonExecutor(qgis_prefix_path=qgis_path)

    # 配置Python路径
    python_path = _get_path_from_config(config, "python_env", "python_path")
    if python_path:
        global local_executor
        local_executor = LocalPythonExecutor(python_path=python_path)

    local_executor.set_show_layer_callback(
        lambda path, layer_name=None: ui_callbacks.get("load_layer", lambda *_: None)(path, layer_name)
    )

@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

    def to_openai_format(self) -> Dict:
        """转换为OpenAI工具格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """获取工具"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict]:
        """列出所有工具（OpenAI格式）"""
        return [tool.to_openai_format() for tool in self.tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """执行工具"""
        tool = self.tools.get(name)
        if not tool:
            return {"error": f"工具不存在: {name}"}
        try:
            return tool.function(**arguments)
        except Exception as e:
            return {"error": str(e)}


# 全局工具注册表
tool_registry = ToolRegistry()


def register_tool(name: str, description: str, parameters: Dict[str, Any]):
    """工具注册装饰器"""
    def decorator(func: Callable):
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=func
        )
        tool_registry.register(tool)
        return func
    return decorator


# ============ 核心执行工具 ============

@register_tool(
    name="run_python_script",
    description="在本地Python环境中执行代码。用于执行数据处理、算法计算等任务。可以使用geopandas, shapely等库。",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的Python代码"
            }
        },
        "required": ["code"]
    }
)
def run_python_script(code: str) -> Dict:
    """执行本地Python代码"""
    result = local_executor.execute(code)
    _dispatch_layer_events(result)
    return result


@register_tool(
    name="run_qgis_script",
    description="在QGIS Python环境中执行代码。用于调用QGIS算法(processing)。需引用 qgis.core 等模块。",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的Python代码"
            }
        },
        "required": ["code"]
    }
)
def run_qgis_script(code: str) -> Dict:
    """执行QGIS Python代码"""
    result = qgis_executor.execute(code)
    _dispatch_layer_events(result)
    return result


@register_tool(
    name="todo_write",
    description="维护结构化Todo List。先写计划，再执行代码。可用于新增、更新、勾选任务状态。",
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "任务项数组，每项包含 task/status/note",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "status": {"type": "string", "enum": ["todo", "doing", "done", "blocked"]},
                        "note": {"type": "string"}
                    },
                    "required": ["task", "status"]
                }
            },
            "replace": {
                "type": "boolean",
                "description": "是否替换当前列表，false为追加",
                "default": True
            }
        },
        "required": ["items"]
    }
)
def todo_write(items: List[Dict[str, Any]], replace: bool = True) -> Dict[str, Any]:
    """写入或更新任务列表"""
    global todo_state
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        task = str(item.get("task", "")).strip()
        status = str(item.get("status", "todo")).strip().lower()
        note = str(item.get("note", "")).strip()
        if not task:
            continue
        if status not in {"todo", "doing", "done", "blocked"}:
            status = "todo"
        normalized.append({"task": task, "status": status, "note": note})

    if replace:
        todo_state = normalized
    else:
        todo_state.extend(normalized)

    return {"status": "success", "todos": todo_state}


@register_tool(
    name="TodoWrite",
    description="`todo_write` 的兼容别名。",
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "status": {"type": "string"},
                        "note": {"type": "string"}
                    },
                    "required": ["task", "status"]
                }
            },
            "replace": {"type": "boolean", "default": True}
        },
        "required": ["items"]
    }
)
def todo_write_alias(items: List[Dict[str, Any]], replace: bool = True) -> Dict[str, Any]:
    return todo_write(items, replace)


# ============ 交互与可视化工具 ============

@register_tool(
    name="load_layer",
    description="加载矢量数据文件到地图",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径（支持shp, geojson, gpkg格式）"
            },
            "layer_name": {
                "type": "string",
                "description": "图层名称（可选）"
            }
        },
        "required": ["file_path"]
    }
)
def load_layer(file_path: str, layer_name: str = None) -> Dict:
    """加载图层"""
    if "load_layer" in ui_callbacks:
        try:
            ui_callbacks["load_layer"](file_path, layer_name)
            return {
                "status": "success",
                "action": "load_layer",
                "file_path": file_path,
                "layer_name": layer_name,
                "message": f"已在地图中加载文件: {file_path}"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    return {
        "status": "warning",
        "message": "UI未连接，无法在地图上显示，但文件路径已确认。",
        "file_path": file_path
    }


@register_tool(
    name="zoom_to_layer",
    description="缩放地图到指定图层范围",
    parameters={
        "type": "object",
        "properties": {
            "layer_name": {
                "type": "string",
                "description": "图层名称"
            }
        },
        "required": ["layer_name"]
    }
)
def zoom_to_layer(layer_name: str) -> Dict:
    """缩放到图层"""
    if "zoom_to_layer" in ui_callbacks:
        try:
            ui_callbacks["zoom_to_layer"](layer_name)
            return {"status": "success", "message": f"已缩放到图层 {layer_name}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    return {"status": "warning", "message": "UI未连接"}


@register_tool(
    name="set_layer_style",
    description="设置图层显示样式",
    parameters={
        "type": "object",
        "properties": {
            "layer_name": {
                "type": "string",
                "description": "图层名称"
            },
            "fill_color": {
                "type": "string",
                "description": "填充颜色（十六进制）",
                "default": "#3388ff"
            },
            "stroke_color": {
                "type": "string",
                "description": "边框颜色",
                "default": "#000000"
            },
            "stroke_width": {
                "type": "number",
                "description": "边框宽度",
                "default": 1.0
            },
            "opacity": {
                "type": "number",
                "description": "透明度(0-1)",
                "default": 0.7
            },
            "point_symbol": {
                "type": "string",
                "description": "点图层符号：circle/square/triangle/diamond/cross/star/plus/hexagon（仅点图层生效）",
                "default": ""
            }
        },
        "required": ["layer_name"]
    }
)
def set_layer_style(layer_name: str, fill_color: str = "#3388ff",
                    stroke_color: str = "#000000", stroke_width: float = 1.0,
                    opacity: float = 0.7, point_symbol: str = "") -> Dict:
    """设置图层样式"""
    if "set_layer_style" in ui_callbacks:
        try:
            style = {
                "fill_color": fill_color,
                "stroke_color": stroke_color,
                "stroke_width": stroke_width,
                "opacity": opacity
            }
            ps = (point_symbol or "").strip()
            if ps:
                style["point_symbol"] = ps
            ui_callbacks["set_layer_style"](layer_name, style)
            return {"status": "success", "message": f"已更新图层 {layer_name} 样式"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    return {"status": "warning", "message": "UI未连接"}


# ============ 文件操作工具 ============

@register_tool(
    name="read_file",
    description="读取文本文件内容",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径"
            }
        },
        "required": ["file_path"]
    }
)
def read_file(file_path: str) -> Dict:
    """读取文件"""
    try:
        if not os.path.exists(file_path):
             return {"status": "error", "message": "文件不存在"}
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            "status": "success",
            "file_path": file_path,
            "content": content[:10000],  # 增加限制长度
            "truncated": len(content) > 10000
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@register_tool(
    name="list_directory",
    description="列出目录内容",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "目录路径"
            }
        },
        "required": ["path"]
    }
)
def list_directory(path: str) -> Dict:
    """列出目录"""
    import os
    try:
        if not os.path.exists(path):
            return {"status": "error", "message": "目录不存在"}
        items = []
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            items.append({
                "name": item,
                "type": "directory" if os.path.isdir(full_path) else "file",
            })
        return {
            "status": "success",
            "path": path,
            "items": items
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============ 技能学习工具 ============

@register_tool(
    name="save_skill",
    description="[学习] 保存当前编写的Python脚本作为一个新的技能，以便下次直接调用。",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "技能名称 (英文，如 calculate_area)"
            },
            "description": {
                "type": "string",
                "description": "技能描述"
            },
            "code": {
                "type": "string",
                "description": "完整的Python代码"
            },
            "usage_example": {
                "type": "string",
                "description": "使用示例 (可选)"
            }
        },
        "required": ["name", "description", "code"]
    }
)
def save_skill(name: str, description: str, code: str, usage_example: str = "") -> Dict:
    """保存技能"""
    return skill_manager.save_skill(name, description, code, usage_example)


@register_tool(
    name="list_skills",
    description="[学习] 列出所有已学习的技能",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    }
)
def list_skills() -> Dict:
    """列出技能"""
    return {"skills": skill_manager.list_skills()}


@register_tool(
    name="get_skill",
    description="[学习] 获取指定技能的代码和详情",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "技能名称"
            }
        },
        "required": ["name"]
    }
)
def get_skill(name: str) -> Dict:
    """获取技能"""
    skill = skill_manager.get_skill(name)
    if skill:
        return {"status": "success", "skill": skill}
    return {"status": "error", "message": "Skill not found"}
