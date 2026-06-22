from __future__ import annotations

from copy import copy
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "POINT_GENERALIZATION_SKILLS.xlsx"

COLUMNS = [
    "Skill名称",
    "Skills类型",
    "点综合对象",
    "综合目标",
    "在点综合中的作用",
    "所属流程阶段",
    "定义文件",
    "调用文件",
    "调用方式",
    "使用状态",
    "证据说明",
    "备注",
]


def row(
    name: str,
    skill_type: str,
    obj: str,
    goal: str,
    role: str,
    stage: str,
    definitions: list[str],
    calls: list[str],
    call_method: str,
    status: str,
    evidence: str,
    notes: str = "",
) -> dict[str, str]:
    return {
        "Skill名称": name,
        "Skills类型": skill_type,
        "点综合对象": obj,
        "综合目标": goal,
        "在点综合中的作用": role,
        "所属流程阶段": stage,
        "定义文件": "\n".join(definitions),
        "调用文件": "\n".join(calls),
        "调用方式": call_method,
        "使用状态": status,
        "证据说明": evidence,
        "备注": notes,
    }


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    rows.append(
        row(
            "point_generalization_skill",
            "符号表达与结果校核类",
            "任意点要素、POI、交通站点、公共服务设施点、监测点",
            "建立点综合总体流程：结构识别、语义识别、算法选择、冲突消解、VLM+LLM 验收。",
            "作为点综合主 skill，约束先继承 map-generalization 的尺度/可读性规则，再执行点专项策略。",
            "流程编排 / prompt 约束 / 验收规范",
            ["skills/point_generalization_skill.md"],
            ["src/agent/core.py:723", "src/agent/core.py:2136", "tests/test_skill_markdown_loading.py"],
            "SkillManager 动态读取 skills/point_generalization_skill.md；MapGenAgent 在点任务 plan.point_strategy.skills 中引用。",
            "已定义且已使用",
            "Markdown 中明确规定点综合 A0-A-D、算法矩阵、统一评分和 VLM+LLM 验收；core.py 在 Selection Basis 和 point_strategy 中直接写入该 skill。",
            "这是本表的主流程 skill，不计入通用 skills 审计。",
        )
    )

    rows.append(
        row(
            "map-generalization",
            "符号表达与结果校核类",
            "点要素在目标比例尺下的符号可读性",
            "提供最小间隔、选择优先于位移、每轮冲突检查等制图综合底层规范。",
            "为点综合提供尺度约束和制图可读性 guardrail。",
            "流程前置约束 / 可读性校核",
            ["src/agent/skills.py", "skills/map_generalization_skill.md"],
            ["skills/point_generalization_skill.md:16", "src/agent/core.py:2137", "tests/test_dynamic_agent_features.py"],
            "BUILTIN_SKILLS 注册；point_generalization_skill 声明必须继承；MapGenAgent 点任务技能列表中引用。",
            "已定义且已使用",
            "point_generalization_skill.md 第 A0 步要求导入 map-generalization 规范；core.py 将 map-generalization 与点 skill 一起写入 point_strategy.skills。",
            "只保留其与点综合直接相关的尺度/可读性部分。",
        )
    )

    rows.append(
        row(
            "point_generalization_skills_dlm_dcm",
            "符号表达与结果校核类",
            "多类别 POI 点",
            "用 DLM/DCM 双层模型组织点综合子技能。",
            "把 DLM 的重要性、结构、类别平衡输出给 DCM 的符号冲突、位移、图面优化环节。",
            "技能蓝图 / workflow 编排",
            ["skills/point_generalization_skills_dlm_dcm.json", "src/skills/point_generalization_dlm_dcm.py:67"],
            ["scripts/run_dlm_dcm_generalization.py:161", "scripts/run_dlm_dcm_generalization.py:163", "tests/test_skill_markdown_loading.py"],
            "build_skill_blueprint() 生成 JSON 蓝图；实验脚本保存到 skills 目录；SkillManager 可按 JSON 读取。",
            "已定义且已使用",
            "JSON 蓝图定义 DLM 与 DCM 两层共 6 个点综合子技能；run_dlm_dcm_generalization.py 调用 build_skill_blueprint 并写入 skills/。",
            "",
        )
    )

    rows.append(
        row(
            "representative_poi_retain",
            "保留与筛选类",
            "多类别 POI 中的代表性点、少数类别点、中心点",
            "构造可解释的重要性得分，优先保留代表点和稀有类别。",
            "计算 category rarity、nearest-neighbor centrality、representative_score、name_score，输出 importance_score、minority_category、cluster_id。",
            "DLM 分析 / 候选点评分",
            ["skills/point_generalization_skills_dlm_dcm.json:9", "src/skills/point_generalization_dlm_dcm.py:76"],
            ["src/skills/point_generalization_dlm_dcm.py:184", "scripts/run_dlm_dcm_generalization.py:254", "scripts/export_final_engine_maps.py:334"],
            "PointGeneralizationEngine.analyze_dlm() 合并实现；实验和出图脚本先调用 analyze_dlm 再进行 DCM 优化。",
            "已定义且已使用",
            "analyze_dlm 中生成 cluster_id、centrality、rarity_score、representative_score、minority_category、importance_score。",
            "子技能名称存在于蓝图，执行逻辑由 engine 方法实现。",
        )
    )

    rows.append(
        row(
            "semantic_importance_selection",
            "保留与筛选类",
            "带等级、权重、类别、线路或名称字段的点",
            "按语义重要性和属性权重选择保留点。",
            "从 importance/priority/weight/level/grade/rank/flow 等数值字段累加 _score，并结合类别稀有度避免单一类型塌缩。",
            "代码生成的 selection 阶段",
            ["skills/point_generalization_skill.md:150", "src/agent/core.py:838"],
            ["src/agent/core.py:838", "src/agent/core.py:881", "src/agent/core.py:2130"],
            "MapGenAgent 生成的点 selection 代码在运行时执行；由 plan.point_strategy 的 semantic_type/structure_type 驱动。",
            "已定义且已使用",
            "point_generalization_skill.md 定义 Score 模板；core.py 在 generic point selection 中实现语义字段评分和类别多样性加分。",
            "属于生成代码中的点专项 skill 单元，未作为 SkillManager 独立条目注册。",
        )
    )

    rows.append(
        row(
            "category_balance_preserve",
            "保留与筛选类",
            "多类别 POI、低频类别、局部网格类别组合",
            "保护局部类别多样性，避免高频类别吞没低频类别。",
            "按 grid_id 为各类别预留代表点，minority 类别提高保留底线，并在 _seed_selection 中标记 guaranteed。",
            "DLM 配额 / 初始种子选择",
            ["skills/point_generalization_skills_dlm_dcm.json:58", "src/skills/point_generalization_dlm_dcm.py:100"],
            ["src/skills/point_generalization_dlm_dcm.py:326", "src/skills/point_generalization_dlm_dcm.py:343", "src/skills/point_generalization_dlm_dcm.py:448"],
            "PointGeneralizationEngine._build_quota() 与 _seed_selection() 内部调用；optimize_dcm() 先取 seeds。",
            "已定义且已使用",
            "_build_quota 根据 preserve_floor、minority_category 和类别频数生成配额；_seed_selection 对多类别网格设置 guaranteed。",
            "",
        )
    )

    rows.append(
        row(
            "region_quota_keep",
            "保留与筛选类",
            "被网格划分的点分布区域、稀疏区域、密集区域",
            "先保区域存在事实，再做热点压缩，避免局部区域整体消失。",
            "按 metric grid 生成 region quota；每个 occupied region 至少一个代表，稀疏区和密集区设不同最低保留。",
            "代码生成的 selection / repair 阶段",
            ["skills/point_generalization_skill.md:102", "src/agent/core.py:974"],
            ["src/agent/core.py:974", "src/agent/core.py:1560", "src/agent/core.py:2010", "tests/test_vlm_verifier.py:107"],
            "MapGenAgent 生成代码在 selection 中执行 region quota pass，并在输出前执行 region coverage repair；VLMVerifier 校核 region coverage。",
            "已定义且已使用",
            "core.py 的 Pass A/Pass B 明确保每个 occupied region 和 dense region extra reps；后处理计算 point_region_coverage_ratio。",
            "对应用户之前强调的“小区域不能整体消失”。",
        )
    )

    rows.append(
        row(
            "hotspot_representative_keep",
            "保留与筛选类",
            "团簇型商业 POI、热点格网、高密单元",
            "保留热点中心和热点轮廓，防止密集区被压缩为空或只剩单点。",
            "识别高频 hotspot cell，先每个热点保高分点，再对最大热点补 x/y 局部锚点。",
            "热点保留 / 选择增强",
            ["skills/point_generalization_skill.md:92", "src/agent/core.py:1187"],
            ["src/agent/core.py:1187", "src/agent/core.py:1970", "tests/test_vlm_verifier.py:78"],
            "MapGenAgent 生成代码的 Dense-hotspot representative pass；指标端计算 point_hotspot_coverage_ratio 和 point_hotspot_rep_count。",
            "已定义且已使用",
            "core.py 中 hotspot_representative_keep 逻辑与 Selection Basis 里的 hotspot_representative_keep 名称一致；VLMVerifier 有热点覆盖/代表点数量规则。",
            "",
        )
    )

    rows.append(
        row(
            "structure_aware_selection",
            "结构保持类",
            "线性、团簇、多中心、稀疏和散点结构",
            "保持点分布的空间结构，不让线性廊道、团簇轮廓或多中心结构塌缩。",
            "DBSCAN 识别聚类；PCA/SVD 判断 linear/clustered；_structure_candidates 保留线性分段骨架、团簇中心和边界锚点。",
            "DLM 结构识别 / 结构候选提取",
            ["skills/point_generalization_skills_dlm_dcm.json:33", "src/skills/point_generalization_dlm_dcm.py:88"],
            ["src/skills/point_generalization_dlm_dcm.py:184", "src/skills/point_generalization_dlm_dcm.py:343", "scripts/render_publication_panels.py:296"],
            "PointGeneralizationEngine.analyze_dlm() 生成 structure_type；_structure_candidates() 将结构转为候选点。",
            "已定义且已使用",
            "代码中使用 DBSCAN、PCA/SVD、cluster centroid、axis projection、anchor ids 生成结构保留候选。",
            "",
        )
    )

    rows.append(
        row(
            "axis_bin_continuity_keep",
            "结构保持类",
            "廊道型、串珠型、交通网络型点分布",
            "保持主轴中段连续性，避免只保两端或中段空洞。",
            "估计主轴方向，将点投影到 axis bins，并按结构/语义类型分配连续性保留 quota。",
            "主轴连续性选择 / 结构校核",
            ["skills/point_generalization_skill.md:133", "src/agent/core.py:1237"],
            ["src/agent/core.py:1237", "src/agent/core.py:1930", "src/agent/vlm_verifier.py:70", "tests/test_vlm_verifier.py:48"],
            "MapGenAgent 生成代码执行 axis bin continuity pass；指标端输出 point_axis_coverage_ratio 与 point_middle_gap_ratio；VLMVerifier 校核阈值。",
            "已定义且已使用",
            "Selection Basis 中记录 axis_bin_continuity_keep；VLMVerifier 测试覆盖 point_middle_gap_rule。",
            "",
        )
    )

    rows.append(
        row(
            "contour_anchor_preserve",
            "结构保持类",
            "点集外轮廓、边界控制点",
            "防止 typification 或筛选后点集外形塌缩。",
            "保留 xmin/xmax/ymin/ymax 与两条对角方向极值，共 8 方向锚点；typification 后若丢失则回补。",
            "轮廓保真 / typification 回补",
            ["skills/point_generalization_skill.md:140", "src/agent/core.py:908"],
            ["src/agent/core.py:908", "src/agent/core.py:1513", "src/skills/point_generalization_dlm_dcm.py:361"],
            "MapGenAgent 生成代码 selection 与 typification 两处执行；PointGeneralizationEngine._structure_candidates 也保留 x/y 极值锚点。",
            "已定义且已使用",
            "point_generalization_skill.md 将轮廓保真列为所有点类型强制约束；core.py 以 anchor_idx 方式实现。",
            "",
        )
    )

    rows.append(
        row(
            "symbol_size_aware_generalization",
            "布局优化与冲突消解类",
            "不同比例尺、不同类别符号尺寸的 POI 点",
            "按符号实际占位判断冲突，而不是按裸点距离判断。",
            "将 symbol_size 和 min_distance 换算为地面米尺度；_conflict_threshold 合成符号半径与最小间距阈值。",
            "DCM 冲突建模 / 符号占位约束",
            ["skills/point_generalization_skills_dlm_dcm.json:86", "src/skills/point_generalization_dlm_dcm.py:117"],
            ["src/skills/point_generalization_dlm_dcm.py:169", "src/skills/point_generalization_dlm_dcm.py:382", "scripts/export_final_multiscale_maps.py:371"],
            "PointGeneralizationEngine.metric_radius_from_rule()、metric_gap_from_rule()、_conflict_threshold() 被冲突计数、位移和 DCM 优化调用。",
            "已定义且已使用",
            "代码中使用 symbol_size * target_scale / 1000 换算半径，按两点符号半径和 gap 形成冲突阈值。",
            "",
        )
    )

    rows.append(
        row(
            "cross_category_conflict_resolution",
            "布局优化与冲突消解类",
            "同类和异类符号冲突点、受保障 minority 类别点",
            "消解符号重叠，同时尽量保护类别多样性。",
            "对冲突点先尝试位移 guaranteed 点；否则按 utility 与 replaceable 规则替换弱候选。",
            "DCM 贪心选择 / 位移消解",
            ["skills/point_generalization_skills_dlm_dcm.json:109", "src/skills/point_generalization_dlm_dcm.py:129"],
            ["src/skills/point_generalization_dlm_dcm.py:448", "src/skills/point_generalization_dlm_dcm.py:414", "scripts/export_multi_region_clean.py:220"],
            "PointGeneralizationEngine.optimize_dcm() 调用 _attempt_displacement()、_utility() 和 _conflict_threshold()。",
            "已定义且已使用",
            "optimize_dcm 中检测 conflict_indices；guaranteed 点尝试局部位移；非保障点按 utility 替换。",
            "",
        )
    )

    rows.append(
        row(
            "layout_aesthetic_optimization",
            "布局优化与冲突消解类",
            "局部空白、局部类别缺失、图面密度不均的综合后 POI",
            "补足缺失类别并改善图面均衡。",
            "在低多样性网格中 backfill missing categories；summarize_metrics 输出 layout_balance_index 和 local_category_diversity。",
            "DCM 图面润色 / 结果评价",
            ["skills/point_generalization_skills_dlm_dcm.json:132", "src/skills/point_generalization_dlm_dcm.py:141"],
            ["src/skills/point_generalization_dlm_dcm.py:505", "src/skills/point_generalization_dlm_dcm.py:533", "scripts/run_dlm_dcm_generalization.py:268"],
            "PointGeneralizationEngine.optimize_dcm() 的 layout polish 执行补点；summarize_metrics() 计算平衡和多样性指标。",
            "已定义且已使用",
            "代码注释为 Layout polish: backfill missing categories in grids with low diversity if conflicts permit。",
            "",
        )
    )

    rows.append(
        row(
            "resolve_conflicts_by_min_separation",
            "布局优化与冲突消解类",
            "运行时生成代码中的点符号冲突",
            "通过最小间隔约束、受保护点优先和像素级 strict gap 删除/保留，降低遮挡。",
            "displacement 阶段先迭代位移，再在 strict_no_overlap 下按 cell quota 和 pixel gap 做筛选。",
            "生成代码 displacement / strict_no_overlap",
            ["skills/point_generalization_skill.md:81", "src/agent/core.py:1374"],
            ["src/agent/core.py:1374", "src/agent/core.py:1438", "src/agent/core.py:736", "src/agent/core.py:1930"],
            "MapGenAgent 生成代码执行 displacement operator；Selection Basis 中记录 selected = resolve_conflicts_by_min_separation(selected)。",
            "已定义且已使用",
            "core.py 使用 min_sep、max_shift、protected flags、strict_pixel_gap 和 strict_density_factor 实现冲突消解。",
            "这是 agent 生成代码链，不是 DLM/DCM engine 内部方法。",
        )
    )

    rows.append(
        row(
            "PointGeneralizationEngine.optimize_dcm",
            "布局优化与冲突消解类",
            "经过 DLM 分析的 metric CRS POI GeoDataFrame",
            "集中执行 DCM 层的选择、冲突判断、位移、替换与图面补点。",
            "从 seeds 开始按 utility 排序，逐点检查冲突，处理 guaranteed 点位移和弱候选替换，最后补缺失类别。",
            "DCM 主执行器",
            ["src/skills/point_generalization_dlm_dcm.py:448"],
            ["scripts/run_dlm_dcm_generalization.py:263", "scripts/export_final_engine_maps.py:360", "scripts/export_multi_region_clean.py:220"],
            "Python import 后实例化 PointGeneralizationEngine，直接调用 engine.optimize_dcm(sample_metric, scale)。",
            "已使用但未集中注册",
            "多个点综合出图/实验脚本直接调用 optimize_dcm；它未作为 SkillManager 独立 skill 注册。",
            "",
        )
    )

    rows.append(
        row(
            "SymbolRule / build_default_symbol_rules",
            "符号表达与结果校核类",
            "八类 POI 及其他类别符号规则",
            "定义符号尺寸、最小距离、优先级、保留底线、颜色和 marker。",
            "为冲突阈值、符号渲染和类别优先级提供统一规则。",
            "符号规则生成 / 符号表达",
            ["src/skills/point_generalization_dlm_dcm.py:27", "src/skills/point_generalization_dlm_dcm.py:47", "src/skills/__init__.py"],
            ["scripts/run_dlm_dcm_generalization.py:342", "scripts/run_dlm_dcm_generalization.py:343", "scripts/render_publication_panels.py:433"],
            "build_default_symbol_rules(categories) 生成规则；PointGeneralizationEngine.save_symbol_rules() 导出 symbol_rules.json；渲染脚本读取 marker/color/size。",
            "已使用但未集中注册",
            "SymbolRule 被 DCM 阈值、legend_handles、plot_map 和冲突可视化共同使用。",
            "此前 ArcGIS 符号需求对应的符号表达基础也来自这里。",
        )
    )

    rows.append(
        row(
            "count_symbol_conflicts",
            "符号表达与结果校核类",
            "综合前后点符号冲突对",
            "量化符号重叠数量，作为综合质量指标。",
            "用 nearest neighbors 找候选对，再按 _conflict_threshold 统计 symbol_conflicts_before/after。",
            "质量校核 / 指标统计",
            ["src/skills/point_generalization_dlm_dcm.py:389"],
            ["src/skills/point_generalization_dlm_dcm.py:595", "scripts/export_final_engine_maps.py:377", "scripts/export_final_multiscale_maps.py:371", "scripts/render_publication_panels.py:209"],
            "PointGeneralizationEngine.count_symbol_conflicts() 被 summarize_metrics 和最终出图脚本调用。",
            "已使用但未集中注册",
            "summarize_metrics 输出 symbol_conflicts_before 与 symbol_conflicts_after；出版图脚本也计算 conflict pairs。",
            "",
        )
    )

    rows.append(
        row(
            "summarize_metrics",
            "符号表达与结果校核类",
            "综合前后 POI 数据集",
            "输出保留率、类别覆盖、局部多样性、少数类别生存率、冲突数和布局均衡度。",
            "作为点综合实验的量化验收表。",
            "结果评价 / 实验汇总",
            ["src/skills/point_generalization_dlm_dcm.py:533"],
            ["scripts/run_dlm_dcm_generalization.py:268", "scripts/run_dlm_dcm_generalization.py:350"],
            "engine.summarize_metrics() 生成 metrics_summary.csv 和 experiment_summary.md 的核心指标。",
            "已使用但未集中注册",
            "指标字段包括 retention_rate、category_coverage_rate、local_category_diversity、symbol_conflicts_before/after、layout_balance_index。",
            "",
        )
    )

    rows.append(
        row(
            "VLMVerifier point QA",
            "符号表达与结果校核类",
            "点综合输出截图、冲突指标、区域覆盖指标、热点指标、主轴连续性指标",
            "模拟/接入 VLM 的点综合验收，判断是否需要继续迭代。",
            "校核 conflict_count_total、retention_rate、point_axis_coverage_ratio、point_hotspot_coverage_ratio、point_region_coverage_ratio、RGB 过密等。",
            "VLM/规则验收 / 迭代停止条件",
            ["src/agent/vlm_verifier.py:13", "src/agent/vlm_verifier.py:30"],
            ["src/agent/vlm_verifier.py:48", "src/agent/vlm_verifier.py:70", "src/agent/vlm_verifier.py:95", "tests/test_vlm_verifier.py"],
            "VLMVerifier.verify() 优先调用外部 callback，否则执行 _heuristic_verify；测试覆盖点中段空洞、热点、区域覆盖和零冲突目标。",
            "已定义且已使用",
            "point_generalization_skill.md 第 6 节要求 VLM+LLM 双模型验收；vlm_verifier.py 中实现多项 point_* 指标规则。",
            "这是校核 skill，不属于通用 VLM 能力泛化审计。",
        )
    )

    rows.append(
        row(
            "Point Generalization Selection Basis report",
            "符号表达与结果校核类",
            "点综合执行记录、profile、operator decision、验证结果、输出文件",
            "生成可解释的点综合选择依据文档。",
            "记录 skill reference、semantic_type、structure_type、operator decision、算法片段、VLM+LLM 验收和输出路径。",
            "解释性报告 / workflow 产物",
            ["src/agent/core.py:712"],
            ["src/agent/core.py:723", "src/agent/core.py:733", "src/agent/core.py:740"],
            "MapGenAgent 在点综合执行后写入 selection_basis markdown。",
            "已定义且已使用",
            "core.py 的 markdown 模板标题为 Point Generalization Selection Basis，并写入 skill reference: skills/point_generalization_skill.md。",
            "用于证明调用链和决策解释，不是独立算法。",
        )
    )

    return rows


def write_excel(rows: list[dict[str, str]]) -> None:
    df = pd.DataFrame(rows, columns=COLUMNS)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Point Skills")
        ws = writer.book["Point Skills"]
        widths = {
            "A": 36,
            "B": 24,
            "C": 34,
            "D": 46,
            "E": 62,
            "F": 28,
            "G": 48,
            "H": 54,
            "I": 54,
            "J": 20,
            "K": 64,
            "L": 42,
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for row_cells in ws.iter_rows():
            for cell in row_cells:
                alignment = copy(cell.alignment)
                alignment.wrap_text = True
                alignment.vertical = "top"
                cell.alignment = alignment


def main() -> None:
    rows = build_rows()
    write_excel(rows)
    df = pd.DataFrame(rows)
    print(f"Wrote {OUTPUT}")
    print(f"Rows: {len(rows)}")
    print(df["Skills类型"].value_counts().to_string())
    print(df["使用状态"].value_counts().to_string())


if __name__ == "__main__":
    main()
