# MapGenAI 智能体开发指南

MapGenAI 集成了一个基于 ReAct 模式的智能体框架，允许用户通过自然语言控制 GIS 操作、编写脚本并学习新算法。

## 1. 架构概览

- **Agent Core (`src/agent/core.py`)**: 负责与 LLM 交互，维护对话历史，解析工具调用。
- **Tools (`src/agent/tools.py`)**: 定义了 Agent 可用的工具集。
- **Executor (`src/agent/executor.py`)**: 负责在不同环境（本地 Python, QGIS Python）中安全执行代码。
- **UI Integration (`src/ui/main_window.py`)**: 通过回调机制将 Agent 与 Qt 界面连接。

## 2. 配置说明

首次使用前，请在"配置 -> AI设置"中完成以下配置：
1. **模型提供商**: 选择 Zhipu (智谱AI), OpenAI, Anthropic 或 Qwen (通义千问)。
2. **API Key**: 输入对应平台的 API 密钥。
3. **QGIS 路径**: 如果需要使用 QGIS 算法，请指定 QGIS 安装根目录 (例如 `C:\Program Files\QGIS 3.34.0`)。

## 3. 工具扩展

要添加新工具，请在 `src/agent/tools.py` 中使用 `@register_tool` 装饰器：

```python
@register_tool(
    name="my_new_tool",
    description="这是一个新工具的描述",
    parameters={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "参数说明"}
        },
        "required": ["param1"]
    }
)
def my_new_tool(param1: str) -> Dict:
    # 实现逻辑
    return {"status": "success", "result": f"Processed {param1}"}
```

## 4. 技能学习 (Skill Learning)

Agent 具备自我学习能力。当 Agent 编写了一个成功的 Python 脚本解决特定问题时，它可以使用 `save_skill` 工具将其保存。
用户也可以显式要求 Agent："把刚才的算法保存为技能 'building_simplification'"。

保存的技能存储在 `skills/` 目录下，格式为 JSON。

## 5. 调试与测试

- 运行 `python tests/verify_qgis.py` 检查 QGIS 环境集成状态。
- 运行 `python tests/test_agent_mock.py` 验证 Agent 核心循环逻辑。

## 6. 常见问题

**Q: Agent 提示 "QGIS环境未找到"**
A: 请在设置中手动指定 QGIS 安装路径。确保路径下包含 `apps/qgis` 或 `bin/qgis_core.dll` 等关键文件。

**Q: 地图操作无反应**
A: 确保 UI 回调已正确注册。Agent 面板会在启动时自动连接到主窗口。
