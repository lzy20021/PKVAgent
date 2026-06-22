import json
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from .knowledge_base import AlgorithmKnowledgeBase
from .providers import Message


CODEGEN_PROMPT = """你是GIS算法代码生成器，请根据任务生成可执行Python代码。

任务描述:
{task_description}

任务类型: {task_type}
算法: {algorithm}
参数:
{parameters}

参考案例:
{examples}

代码模板骨架:
{template}

要求:
1. 使用GeoPandas/Shapely或QGIS API
2. 包含try/except错误处理
3. 涉及距离或面积时检查并处理CRS
4. 调用show_layer(output_file, layer_name)
5. 打印关键执行日志
仅输出Python代码。
"""


class GenerationStrategy(str, Enum):
    TEMPLATE = "template"
    FEW_SHOT = "few_shot"
    HYBRID = "hybrid"


class CodeGenerator:
    def __init__(self, llm_provider: Any = None, knowledge_base: Optional[AlgorithmKnowledgeBase] = None):
        self.llm = llm_provider
        self.kb = knowledge_base or AlgorithmKnowledgeBase()
        self._cache: Dict[str, str] = {}

    def generate(self, task: Dict[str, Any], strategy: Optional[str] = None) -> str:
        strategy = (strategy or "").lower().strip()
        cache_key = self._generate_cache_key(task, strategy)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if strategy == GenerationStrategy.TEMPLATE.value:
            code = self._template_fill(task)
        elif strategy == GenerationStrategy.FEW_SHOT.value:
            code = self._few_shot_generate(task)
        elif strategy == GenerationStrategy.HYBRID.value:
            code = self._hybrid_generate(task)
        else:
            algorithm = (task.get("algorithm") or task.get("parameters", {}).get("algorithm") or "").strip()
            if algorithm and self.kb.get_algorithm(algorithm):
                code = self._template_fill(task)
            else:
                code = self._hybrid_generate(task)

        self._cache[cache_key] = code
        return code

    def _generate_cache_key(self, task: Dict[str, Any], strategy: str = "") -> str:
        payload = {
            "strategy": strategy,
            "task_type": task.get("task_type"),
            "algorithm": task.get("algorithm") or task.get("parameters", {}).get("algorithm"),
            "parameters": task.get("parameters", {}),
            "raw_input": task.get("raw_input", ""),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def _template_fill(self, task: Dict[str, Any]) -> str:
        algorithm_name = (task.get("algorithm") or task.get("parameters", {}).get("algorithm") or "").strip()
        if not algorithm_name:
            # fallback to best matched algorithm by task_type
            candidates = self.kb.search(task_type=task.get("task_type"), keywords=task.get("keywords", []), limit=1)
            if not candidates:
                raise ValueError("无法匹配模板算法")
            algo = candidates[0]
        else:
            algo = self.kb.get_algorithm(algorithm_name)
            if not algo:
                raise ValueError(f"算法不存在: {algorithm_name}")

        template = str(algo.get("code_template", ""))
        if not template:
            raise ValueError("算法模板为空")

        params = self._prepare_params(task.get("parameters", {}))
        try:
            # 安全检查：仅填充模板中存在的键
            # 有些模板可能没有使用所有参数，直接 format 会报错吗？不会，多余参数不影响
            # 但是如果模板里有 {key} 而 params 里没有，就会报错
            # 所以我们需要确保 params 包含了模板所需的所有 key
            # 这里简单处理：捕获 KeyError 并抛出更友好的错误
            # 也可以先分析模板需要的 keys，再从 params 里取
            # 或者使用 safe_substitute (但那是 string.Template)
            return template.format(**params)
        except KeyError as exc:
            # 针对 len(gdf) 这种非参数的模板变量，它是运行时计算的，不应该在生成阶段替换
            # 这是一个模板设计问题。模板里不应该包含运行时才有的变量作为占位符
            # 除非我们在 params 里预注入了这些变量的字面量（比如 "len(gdf)"）
            # 但这里 douglas_peucker.json 里的模板写了 {len(gdf)}
            # 这意味着它期望 len(gdf) 是一个被替换的值
            # 但实际上 len(gdf) 应该是代码的一部分，而不是生成时替换的变量
            # 修正方案：将模板中的 {len(gdf)} 改为 {{len(gdf)}} 以转义，或者在 params 里提供伪值
            # 鉴于我们无法修改所有 json 模板，我们在代码里做一个临时补丁：
            # 如果 key 看起来像代码表达式（包含括号），我们尝试保留它
            
            # 但更好的方法是：修改 _prepare_params，注入常见的运行时表达式占位符
            # 或者，修改 json 模板，把 {len(gdf)} 变成 {{len(gdf)}}
            # 由于我不能修改 json 文件（数量多且易错），我在 _prepare_params 里加一个 hack
            raise ValueError(f"缺少必需参数: {exc}")

    def _few_shot_generate(self, task: Dict[str, Any]) -> str:
        if not self.llm:
            return self._generic_fallback(task)

        examples = self._select_examples(task, limit=3)
        template_stub = self._template_stub(task)
        prompt = self._build_prompt(task, examples, template_stub)

        try:
            response = self.llm.chat([Message("user", prompt)])
            return self._extract_code(response.content or "")
        except Exception:
            return self._generic_fallback(task)

    def _hybrid_generate(self, task: Dict[str, Any]) -> str:
        try:
            base = self._template_fill(task)
        except Exception:
            return self._few_shot_generate(task)

        if not self.llm:
            return base

        examples = self._select_examples(task, limit=2)
        refine_prompt = (
            "你将获得一段GIS代码骨架。请在不改变核心参数含义的前提下完善异常处理和日志输出。"
            "仅输出Python代码。\n\n"
            f"任务: {task.get('raw_input', task.get('description', ''))}\n"
            f"参考案例: {json.dumps(examples, ensure_ascii=False)}\n"
            f"代码骨架:\n```python\n{base}\n```"
        )
        try:
            response = self.llm.chat([Message("user", refine_prompt)])
            refined = self._extract_code(response.content or "")
            return refined or base
        except Exception:
            return base

    def _prepare_params(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(raw_params or {})

        if "input_file" not in params:
            params["input_file"] = "input.geojson"

        if "output_file" not in params:
            input_file = str(params["input_file"])
            params["output_file"] = re.sub(
                r"\.(shp|geojson|gpkg)$",
                r"_result.\1",
                input_file,
                flags=re.IGNORECASE,
            )
            if params["output_file"] == input_file:
                params["output_file"] = f"{input_file}_result.geojson"

        if "layer_name" not in params:
            params["layer_name"] = os.path.splitext(os.path.basename(str(params["output_file"])))[0] or "result_layer"

        params.setdefault("tolerance", 10)
        params.setdefault("distance", 100)
        params.setdefault("target_crs", "EPSG:3857")
        params.setdefault("preserve_topology", True)
        params.setdefault("by_field", "group")
        params.setdefault("aggfunc", "first")
        params.setdefault("resolution", 8)
        params.setdefault("clip_layer", "data/study_area.shp")
        params.setdefault("overlay_layer", "data/overlay.shp")
        params.setdefault("join_layer", "data/join_layer.shp")
        params.setdefault("how", "intersection")
        params.setdefault("predicate", "intersects")

        # Hack for templates using runtime expressions as placeholders
        # e.g. {len(gdf)} in douglas_peucker.json
        # This should ideally be fixed in the JSON templates by escaping braces like {{len(gdf)}}
        params.setdefault("len(gdf)", "{len(gdf)}")
        params.setdefault("input_file", "input.shp")  # Fallback
        
        return params

    def _select_examples(self, task: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
        task_type = task.get("task_type")
        algorithm = task.get("algorithm") or task.get("parameters", {}).get("algorithm")
        algorithms = self.kb.search(task_type=task_type, keywords=task.get("keywords", []), limit=20)

        examples: List[Dict[str, Any]] = []
        if algorithm:
            for algo in algorithms:
                if algo.get("algorithm") == algorithm:
                    examples.extend(algo.get("examples", []))
        if len(examples) < limit:
            for algo in algorithms:
                if not algorithm or algo.get("algorithm") != algorithm:
                    examples.extend(algo.get("examples", []))
                if len(examples) >= limit:
                    break

        return examples[:limit]

    def _template_stub(self, task: Dict[str, Any]) -> str:
        try:
            return self._template_fill(task)
        except Exception:
            return ""

    def _build_prompt(self, task: Dict[str, Any], examples: List[Dict[str, Any]], template: str) -> str:
        return CODEGEN_PROMPT.format(
            task_description=task.get("raw_input", task.get("description", "")),
            task_type=task.get("task_type", "unknown"),
            algorithm=task.get("algorithm") or task.get("parameters", {}).get("algorithm", "auto"),
            parameters=json.dumps(task.get("parameters", {}), ensure_ascii=False, indent=2),
            examples=json.dumps(examples, ensure_ascii=False, indent=2),
            template=template,
        )

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _generic_fallback(self, task: Dict[str, Any]) -> str:
        p = self._prepare_params(task.get("parameters", {}))
        return (
            "import geopandas as gpd\n"
            "try:\n"
            f"    gdf = gpd.read_file(r\"{p['input_file']}\")\n"
            "    if getattr(gdf, 'crs', None) is not None and gdf.crs.is_geographic:\n"
            "        gdf = gdf.to_crs('EPSG:3857')\n"
            f"    gdf.to_file(r\"{p['output_file']}\")\n"
            f"    show_layer(r\"{p['output_file']}\", \"{p['layer_name']}\")\n"
            "    print('done')\n"
            "except Exception as e:\n"
            "    print(f'error: {e}')\n"
            "    raise\n"
        )
