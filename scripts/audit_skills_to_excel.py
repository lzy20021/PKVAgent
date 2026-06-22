from __future__ import annotations

import json
import os
import re
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SKILLS_AUDIT.xlsx"

COLUMNS = [
    "Skill名称",
    "类别",
    "定义文件",
    "调用文件",
    "调用方式",
    "主要作用",
    "使用状态",
    "备注",
]

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".toml",
}

EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "output",
    "outputs",
    "tmp",
    ".next",
    ".turbo",
}

SCAN_ROOTS = [
    "skills",
    "src",
    "scripts",
    "examples",
    "tests",
    "knowledge_base",
    "openclaw-main/src/agents/skills",
    "openclaw-main/ui/src/ui/controllers",
    "openclaw-main/ui/src/ui/views",
    "openclaw-main/skills",
    "openclaw-main/.agents",
    "openclaw-main/extensions/feishu",
    "openclaw-main/extensions/lobster",
    "openclaw-main/extensions/open-prose",
]

OPENCLAW_LOADER_FILES = [
    "openclaw-main/src/agents/skills/workspace.ts",
    "openclaw-main/src/agents/skills/bundled-context.ts",
    "openclaw-main/src/agents/skills/bundled-dir.ts",
    "openclaw-main/src/agents/skills/config.ts",
    "openclaw-main/src/agents/skills/frontmatter.ts",
    "openclaw-main/src/agents/skills/plugin-skills.ts",
]


@dataclass
class SkillRecord:
    name: str
    category: str
    definition_files: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    purpose: str = ""
    call_files: set[str] = field(default_factory=set)
    call_method: str = ""
    status: str = "已定义未使用"
    notes: str = ""


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def is_text_path(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    parts = set(path.relative_to(ROOT).parts)
    if parts & EXCLUDED_PARTS:
        return False
    return True


def iter_text_files() -> Iterable[Path]:
    root_files = [
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "package.json",
        ROOT / "openclaw-main/AGENTS.md",
        ROOT / "openclaw-main/package.json",
    ]
    for path in root_files:
        if path.exists() and is_text_path(path):
            yield path

    for scan_root in SCAN_ROOTS:
        base = ROOT / scan_root
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            current = Path(dirpath)
            dirnames[:] = [
                d
                for d in dirnames
                if d not in EXCLUDED_PARTS and d.lower() not in EXCLUDED_PARTS
            ]
            if set(current.relative_to(ROOT).parts) & EXCLUDED_PARTS:
                continue
            for filename in filenames:
                path = current / filename
                if is_text_path(path):
                    yield path


def parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    data: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("\"'")
    return data


def first_heading_or_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if not s or s == "---":
            continue
        if s.startswith("#"):
            return s.lstrip("#").strip()
        if not re.match(r"^[A-Za-z_-]+:\s*", s):
            return s[:160]
    return ""


def add_record(records: dict[str, SkillRecord], record: SkillRecord) -> SkillRecord:
    key = record.name
    if key in records:
        target = records[key]
        target.definition_files |= record.definition_files
        target.aliases |= record.aliases
        if not target.purpose and record.purpose:
            target.purpose = record.purpose
        if target.category != record.category and record.category not in target.category:
            target.category = f"{target.category}; {record.category}"
        return target
    records[key] = record
    return record


def collect_project_skill_definitions(records: dict[str, SkillRecord]) -> None:
    skills_py = ROOT / "src/agent/skills.py"
    if skills_py.exists():
        text = read_text(skills_py)
        for name in re.findall(r'"([^"]+)"\s*:\s*Skill\(', text):
            add_record(
                records,
                SkillRecord(
                    name=name,
                    category="集中注册内置 skill",
                    definition_files={rel(skills_py)},
                    aliases={name, name.replace("-", "_")},
                    purpose="SkillManager.BUILTIN_SKILLS 中注册的内置制图综合能力。",
                    notes="集中注册入口为 src/agent/skills.py。",
                ),
            )

    for path in (ROOT / "skills").glob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".json"}:
            continue
        stem = path.stem
        text = read_text(path)
        purpose = ""
        aliases = {stem, stem.replace("_", "-")}
        category = "项目 skills 目录声明"
        if path.suffix.lower() == ".md":
            fm = parse_frontmatter(text)
            purpose = fm.get("description") or first_heading_or_line(text)
        else:
            try:
                data = json.loads(text)
                purpose = data.get("description") or data.get("purpose") or data.get("name", "")
                if data.get("name"):
                    aliases.add(str(data["name"]))
                category = "项目 JSON skill 蓝图"
            except json.JSONDecodeError:
                purpose = "JSON skill definition"

        add_record(
            records,
            SkillRecord(
                name=stem,
                category=category,
                definition_files={rel(path)},
                aliases=aliases,
                purpose=purpose,
                notes="SkillManager 可按 skills/<name>.md 或 skills/<name>.json 动态读取。",
            ),
        )


def collect_src_skill_modules(records: dict[str, SkillRecord]) -> None:
    src_skills = ROOT / "src/skills"
    exported = set()
    init_path = src_skills / "__init__.py"
    if init_path.exists():
        exported |= set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", read_text(init_path)))

    for path in src_skills.glob("*.py"):
        if path.name == "__init__.py":
            continue
        text = read_text(path)
        class_names = re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.M)
        func_names = re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.M)
        for name in class_names + [f for f in func_names if not f.startswith("_")]:
            if name not in exported and name not in {
                "build_default_symbol_rules",
                "build_skill_blueprint",
            }:
                continue
            purpose = {
                "StyleRenderer": "根据任务、对象类型和尺度生成渲染样式参数。",
                "PerformanceOptimizer": "根据运行历史估算复杂度、优化执行方案和记录性能指标。",
                "PointGeneralizationEngine": "执行点状 POI 的 DLM/DCM 综合选择、冲突检测、位移与指标汇总。",
                "SymbolRule": "描述点符号尺寸、优先级、标签和配色规则。",
                "build_default_symbol_rules": "生成八类 POI 的默认符号规则。",
                "build_skill_blueprint": "生成点综合 DLM/DCM skill 蓝图 JSON。",
            }.get(name, f"{path.stem} 模块导出的能力。")
            add_record(
                records,
                SkillRecord(
                    name=name,
                    category="Python 能力模块",
                    definition_files={rel(path), rel(init_path) if init_path.exists() else ""},
                    aliases={name, name.replace("_", "-")},
                    purpose=purpose,
                    notes="通过 src/skills/__init__.py 导出，未纳入 SkillManager.BUILTIN_SKILLS 集中注册。",
                ),
            )


def collect_dlm_dcm_subskills(records: dict[str, SkillRecord]) -> None:
    json_path = ROOT / "skills/point_generalization_skills_dlm_dcm.json"
    source_path = ROOT / "src/skills/point_generalization_dlm_dcm.py"
    if not json_path.exists():
        return
    data = json.loads(read_text(json_path))
    groups: list[tuple[str, list[dict]]] = []
    for layer_name in ("DLM", "DCM"):
        layer = data.get("layers", {}).get(layer_name, {})
        groups.append((layer_name, layer.get("skills", [])))
    groups.extend(
        [
            ("DLM", data.get("dlm_skills", [])),
            ("DCM", data.get("dcm_skills", [])),
        ]
    )
    for layer_name, items in groups:
        for item in items:
            name = item.get("id") or item.get("name")
            if not name:
                continue
            add_record(
                records,
                SkillRecord(
                    name=name,
                    category=f"点综合 {layer_name} 子技能",
                    definition_files={rel(json_path), rel(source_path)},
                    aliases={name, name.replace("_", "-")},
                    purpose=item.get("description", ""),
                    call_method="PointGeneralizationEngine 合并实现",
                    notes="子技能名主要存在于蓝图；实际行为由 PointGeneralizationEngine 的分析、选择、冲突消解和评价方法组合实现。",
                ),
            )


def collect_kb_algorithms(records: dict[str, SkillRecord]) -> None:
    kb_root = ROOT / "knowledge_base/algorithms"
    for path in kb_root.rglob("*.json"):
        try:
            data = json.loads(read_text(path))
        except json.JSONDecodeError:
            continue
        name = data.get("algorithm") or data.get("name") or path.stem
        aliases = {str(name), path.stem, path.stem.replace("_", "-")}
        if data.get("name"):
            aliases.add(str(data["name"]))
        add_record(
            records,
            SkillRecord(
                name=str(name),
                category="知识库算法能力模块",
                definition_files={rel(path), "src/agent/knowledge_base.py"},
                aliases=aliases,
                purpose=str(data.get("description") or data.get("summary") or data.get("name") or ""),
                notes="由 AlgorithmKnowledgeBase 加载，并由 task_classifier/code_generator 按任务与算法名检索使用。",
            ),
        )


def collect_tool_registry(records: dict[str, SkillRecord]) -> None:
    tools_path = ROOT / "src/agent/tools.py"
    for name, purpose in {
        "save_skill": "将用户提供的 skill 保存到 skills 目录。",
        "list_skills": "列出 SkillManager 当前可用的内置与文件型 skills。",
        "get_skill": "按名称读取 SkillManager 的内置或文件型 skill 内容。",
    }.items():
        add_record(
            records,
            SkillRecord(
                name=name,
                category="tool registry skill 管理工具",
                definition_files={rel(tools_path), "src/agent/skills.py"},
                aliases={name, name.replace("_", "-")},
                purpose=purpose,
                notes="这是工具注册表中的 skill 管理工具，不是地图综合业务 skill。",
            ),
        )


def collect_openclaw_skill_definitions(records: dict[str, SkillRecord]) -> None:
    roots = [
        ROOT / "openclaw-main/skills",
        ROOT / "openclaw-main/.agents/skills",
        ROOT / "openclaw-main/extensions",
    ]
    for base in roots:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            current = Path(dirpath)
            dirnames[:] = [
                d
                for d in dirnames
                if d not in EXCLUDED_PARTS and d.lower() not in EXCLUDED_PARTS
            ]
            if "SKILL.md" not in filenames:
                continue
            path = current / "SKILL.md"
            text = read_text(path)
            fm = parse_frontmatter(text)
            name = fm.get("name") or path.parent.name
            description = fm.get("description") or first_heading_or_line(text)
            if "/skills/" in rel(path) or "\\skills\\" in str(path):
                category = "OpenClaw bundled/plugin skill"
            elif "/.agents/skills/" in rel(path):
                category = "OpenClaw workspace skill"
            else:
                category = "OpenClaw skill"
            add_record(
                records,
                SkillRecord(
                    name=name,
                    category=category,
                    definition_files={rel(path)},
                    aliases={name, name.replace("-", "_")},
                    purpose=description,
                    notes="OpenClaw 子项目的 SKILL.md 定义；由 OpenClaw skill loader 扫描后进入 prompt/context 或命令规格。",
                ),
            )


def build_corpus() -> list[tuple[Path, str]]:
    corpus = []
    for path in iter_text_files():
        rp = rel(path)
        if rp == "scripts/audit_skills_to_excel.py":
            continue
        corpus.append((path, read_text(path)))
    return corpus


def line_refs_for_pattern(path: Path, text: str, pattern: re.Pattern[str], limit: int = 5) -> list[str]:
    refs = []
    for i, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            refs.append(f"{rel(path)}:{i}")
            if len(refs) >= limit:
                break
    return refs


def find_calls(records: dict[str, SkillRecord], corpus: list[tuple[Path, str]]) -> None:
    for record in records.values():
        aliases = sorted({a for a in record.aliases | {record.name} if a and len(a) > 1}, key=len, reverse=True)
        if not aliases:
            continue
        pattern = re.compile("|".join(re.escape(a) for a in aliases), flags=re.I)
        for path, text in corpus:
            rp = rel(path)
            if rp in record.definition_files:
                continue
            refs = line_refs_for_pattern(path, text, pattern, limit=2)
            if refs:
                record.call_files.update(refs)


def enrich_usage(records: dict[str, SkillRecord]) -> None:
    # Explicitly mark dynamic loaders and call chains that are real usage but not always name-based.
    for name in ("point_generalization_skill", "point_generalization_skills_dlm_dcm"):
        if name in records:
            records[name].call_files.update(
                {
                    "src/agent/skills.py:58",
                    "src/agent/skills.py:86",
                    "tests/test_skill_markdown_loading.py:13",
                }
            )
            records[name].call_method = "SkillManager 按 skills/<name>.md/json 动态读取；agent prompt/test 直接引用"

    if "map-generalization" in records:
        records["map-generalization"].definition_files.add("skills/map_generalization_skill.md")
        records["map-generalization"].call_files.update(
            {
                "src/agent/core.py:29",
                "src/agent/vlm_verifier.py:21",
                "tests/test_dynamic_agent_features.py:23",
            }
        )
        records["map-generalization"].call_method = "BUILTIN_SKILLS 注册；系统提示词和测试按名称调用"

    engine_users = {
        "scripts/export_final_engine_maps.py",
        "scripts/export_final_multiscale_maps.py",
        "scripts/export_multi_region_clean.py",
        "scripts/refine_engine_zoom_maps.py",
        "scripts/render_publication_panels.py",
        "scripts/run_dlm_dcm_generalization.py",
    }
    for name in (
        "PointGeneralizationEngine",
        "SymbolRule",
        "build_default_symbol_rules",
        "build_skill_blueprint",
    ):
        if name in records:
            records[name].call_files.update(engine_users)
            records[name].call_method = "Python import / 实例化 / 函数调用"

    for name in ("StyleRenderer", "PerformanceOptimizer"):
        if name in records:
            records[name].call_files.add("examples/skill_demo.py")
            records[name].call_method = "Python import / 示例调用"

    for name in ("save_skill", "list_skills", "get_skill"):
        if name in records:
            records[name].call_files.update(
                {
                    "src/agent/tools.py",
                    "tests/test_tools_events.py",
                    "tests/test_dynamic_agent_features.py",
                    "tests/test_skill_markdown_loading.py",
                }
            )
            records[name].call_method = "工具注册表函数调用 SkillManager"

    kb_callers = {
        "src/agent/knowledge_base.py",
        "src/agent/code_generator.py",
        "src/agent/task_classifier.py",
        "tests/test_knowledge_base.py",
        "tests/test_task_classifier.py",
        "tests/test_code_generator.py",
    }
    for record in records.values():
        if "知识库算法能力模块" in record.category:
            record.call_files.update(kb_callers)
            record.call_method = "AlgorithmKnowledgeBase 加载/检索；TaskClassifier 与 CodeGenerator 使用算法名"

    openclaw_loader_refs = {f for f in OPENCLAW_LOADER_FILES if (ROOT / f).exists()}
    for record in records.values():
        if record.category.startswith("OpenClaw"):
            record.call_files.update(openclaw_loader_refs)
            if record.name == "1password":
                record.call_files.add("openclaw-main/AGENTS.md:184")
            record.call_method = "OpenClaw skill loader 扫描 SKILL.md；构造 prompt/context 或 command specs"

    dcm_engine_refs = {
        "src/skills/point_generalization_dlm_dcm.py",
        "scripts/run_dlm_dcm_generalization.py",
        "scripts/export_final_engine_maps.py",
    }
    for record in records.values():
        if record.category.startswith("点综合 DLM") or record.category.startswith("点综合 DCM"):
            record.call_files.update(dcm_engine_refs)
            if not record.call_method:
                record.call_method = "PointGeneralizationEngine 内部调用链实现"


def classify_status(records: dict[str, SkillRecord]) -> None:
    central_names = {"map-generalization", "point_generalization_skill", "point_generalization_skills_dlm_dcm"}
    for record in records.values():
        called = bool(record.call_files)
        if not called:
            record.status = "已定义未使用"
            record.call_method = record.call_method or "未发现实际调用"
            continue
        if (
            record.category == "Python 能力模块"
            or record.name in {"StyleRenderer", "PerformanceOptimizer", "PointGeneralizationEngine", "SymbolRule"}
        ):
            record.status = "已使用但未集中注册"
            if "未纳入 SkillManager.BUILTIN_SKILLS" not in record.notes:
                record.notes = (record.notes + " 未纳入 SkillManager.BUILTIN_SKILLS。").strip()
        elif record.name in central_names or "集中注册" in record.category or "项目" in record.category:
            record.status = "已定义且已使用"
        elif "OpenClaw" in record.category:
            record.status = "已定义且已使用"
            record.notes = (
                record.notes
                + " 对未被 AGENTS.md 直接点名的条目，实际使用依据为 OpenClaw loader 对 SKILL.md 的集中扫描。"
            ).strip()
        else:
            record.status = "已定义且已使用"
        if not record.call_method:
            record.call_method = "名称引用 / 注册表加载 / 配置扫描"


def rows_from_records(records: dict[str, SkillRecord]) -> list[dict[str, str]]:
    rows = []
    for record in sorted(records.values(), key=lambda r: (r.category, r.name.lower())):
        definition_files = sorted(f for f in record.definition_files if f)
        call_files = sorted(record.call_files)
        rows.append(
            {
                "Skill名称": record.name,
                "类别": record.category,
                "定义文件": "\n".join(definition_files),
                "调用文件": "\n".join(call_files[:24]) + ("\n..." if len(call_files) > 24 else ""),
                "调用方式": record.call_method,
                "主要作用": record.purpose,
                "使用状态": record.status,
                "备注": record.notes,
            }
        )
    return rows


def write_excel(rows: list[dict[str, str]]) -> None:
    df = pd.DataFrame(rows, columns=COLUMNS)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Skills Audit")
        ws = writer.book["Skills Audit"]
        widths = {
            "A": 32,
            "B": 28,
            "C": 48,
            "D": 62,
            "E": 42,
            "F": 58,
            "G": 20,
            "H": 58,
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for row in ws.iter_rows():
            for cell in row:
                alignment = copy(cell.alignment)
                alignment.wrap_text = True
                alignment.vertical = "top"
                cell.alignment = alignment


def main() -> None:
    records: dict[str, SkillRecord] = {}
    collect_project_skill_definitions(records)
    collect_src_skill_modules(records)
    collect_dlm_dcm_subskills(records)
    collect_kb_algorithms(records)
    collect_tool_registry(records)
    collect_openclaw_skill_definitions(records)

    corpus = build_corpus()
    find_calls(records, corpus)
    enrich_usage(records)
    classify_status(records)
    rows = rows_from_records(records)
    write_excel(rows)

    status_counts = pd.Series([row["使用状态"] for row in rows]).value_counts().to_dict()
    print(f"Wrote {OUTPUT}")
    print(f"Rows: {len(rows)}")
    print("Status counts:", status_counts)


if __name__ == "__main__":
    main()
