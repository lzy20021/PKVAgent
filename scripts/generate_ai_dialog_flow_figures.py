from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_ui_rework"
PAPER_DIR = Path(r"E:\GradOpening\论文\软件图\UI重绘")


def tspan_lines(lines: list[str], x: int, y: int, cls: str, gap: int = 22) -> str:
    return "\n".join(f'<text class="{cls}" x="{x}" y="{y + i * gap}">{line}</text>' for i, line in enumerate(lines))


def flow_svg(lang: str) -> str:
    zh = lang == "zh"
    if zh:
        assistant = "AI 助手"
        clear = "清空"
        request = "综合至 1:50,000"
        done = "已完成"
        mapgen = "MapGenAI"
        cards = [
            ("1", "任务解析", ["目标比例尺", "符号保留约束"]),
            ("2", "数据评估", ["类别与密度", "空间分布格局"]),
            ("3", "策略选择", ["匹配规则库", "确定综合算子"]),
            ("4", "迭代执行", ["调用 GIS 工具", "记录运行状态"]),
            ("5", "质量复核", ["检查符号与冲突", "调整参数阈值"]),
            ("6", "结果汇总", ["导出图层", "生成方法摘要"]),
        ]
        progress = [
            ("数据类型", "商业 POI / 八类专题符号"),
            ("空间格局", "中心-外围结构，局部高密度聚集"),
            ("目标尺度", "1:50,000"),
            ("候选算子", "选取、简化、典型化、位移"),
        ]
        tools = [("run_python_script", "完成 5.8s"), ("run_python_script", "完成 3.7s"), ("symbol_check", "通过"), ("export_figures", "已导出")]
        summary_title = "综合结果摘要"
        summary_lines = ["方法：数据评估驱动的多算子综合。", "结果：八类符号保持可区分。"]
        input_hint = "向 MapGenAI 输入综合需求..."
    else:
        assistant = "AI Assistant"
        clear = "Clear"
        request = "Generalize to 1:50,000"
        done = "Done"
        mapgen = "MapGenAI"
        cards = [
            ("1", "Task Parsing", ["Target scale", "Symbol constraints"]),
            ("2", "Data Review", ["Categories and density", "Spatial pattern"]),
            ("3", "Strategy", ["Match rule base", "Select operators"]),
            ("4", "Execution", ["Call GIS tools", "Log run state"]),
            ("5", "Quality Check", ["Symbols and conflicts", "Tune thresholds"]),
            ("6", "Summary", ["Export layer", "Write method note"]),
        ]
        progress = [
            ("Data type", "Commercial POI / eight symbols"),
            ("Spatial pattern", "Center-periphery clusters"),
            ("Target scale", "1:50,000"),
            ("Operators", "Select, simplify, typify, displace"),
        ]
        tools = [("run_python_script", "Done 5.8s"), ("run_python_script", "Done 3.7s"), ("symbol_check", "Passed"), ("export_figures", "Exported")]
        summary_title = "Generalization Summary"
        summary_lines = ["Method: data-aware multi-operator workflow.", "Result: distinguishable POI symbols."]
        input_hint = "Ask MapGenAI to generalize this layer..."

    card_svg = []
    positions = [(430, 74), (670, 74), (910, 74), (430, 294), (670, 294), (910, 294)]
    for (num, head, lines), (x, y) in zip(cards, positions):
        card_svg.append(f"""
<rect class="flow-card" x="{x}" y="{y}" width="210" height="132" rx="8"/>
<circle cx="{x + 30}" cy="{y + 34}" r="17" fill="#e8f3ff" stroke="#a9d0f2"/>
<text class="num" x="{x + 25}" y="{y + 40}">{num}</text>
<text class="card-title" x="{x + 58}" y="{y + 40}">{head}</text>
{tspan_lines(lines, x + 58, y + 78, "card-text", 24)}
""")

    arrows = """
<path class="arrow" d="M640 140 H660"/>
<path class="arrow" d="M880 140 H900"/>
<path class="arrow" d="M1015 206 V258 C1015 282 998 294 980 294"/>
<path class="arrow" d="M910 360 H890"/>
<path class="arrow" d="M670 360 H650"/>
"""

    progress_rows = []
    for i, (k, v) in enumerate(progress):
        y = 320 + i * 26
        progress_rows.append(f'<text class="key" x="70" y="{y}">{k}</text><text class="value" x="162" y="{y}">{v}</text>')

    tool_rows = []
    for i, (name, state) in enumerate(tools):
        y = 452 + i * 40
        tool_rows.append(f"""
<rect x="54" y="{y}" width="294" height="31" rx="7" fill="#fff" stroke="#d9e1ea"/>
<circle cx="74" cy="{y + 16}" r="5" fill="#079669"/>
<text class="tool-name" x="94" y="{y + 21}">Calling {name}</text>
<rect x="242" y="{y + 6}" width="88" height="19" rx="10" fill="#c9f7df"/>
<text class="tool-state" x="264" y="{y + 20}">{state}</text>
""")

    summary_y = 612
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720">
<defs>
  <marker id="arrowHead" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="8" markerHeight="8" orient="auto">
    <path d="M0 0 L10 5 L0 10z" fill="#667085"/>
  </marker>
  <filter id="shadow" x="-8%" y="-8%" width="116%" height="124%">
    <feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#101828" flood-opacity=".12"/>
  </filter>
  <style>
    text{{font-family:"Microsoft YaHei","PingFang SC","Segoe UI",Arial,sans-serif;letter-spacing:0;fill:#17202a}}
    .title{{font-size:28px;font-weight:800}} .subtitle{{font-size:14px;fill:#667085}}
    .panel{{fill:#fff;stroke:#d9e1ea;stroke-width:1.2;filter:url(#shadow)}} .ui-line{{stroke:#e3e8ef;stroke-width:1}}
    .chat-title{{font-size:16px;font-weight:800}} .clear{{font-size:13px;fill:#2563eb}}
    .request{{font-size:14px;fill:#0b3b8c}} .assistant{{font-size:14px;font-weight:800;fill:#344054}}
    .done{{font-size:14px;font-weight:800;fill:#047857}} .reason-title{{font-size:13px;font-weight:800;fill:#344054}}
    .key{{font-size:11px;font-weight:800;fill:#526071}} .value{{font-size:11px;fill:#667085}}
    .check{{font-size:14px;font-weight:800;fill:#079669}} .tool-name{{font-size:11.5px;font-weight:700;fill:#344054}} .tool-state{{font-size:10.5px;fill:#047857}}
    .summary-title{{font-size:13px;font-weight:800;fill:#0369a1}} .summary{{font-size:11.5px;fill:#075985}}
    .input{{font-size:12px;fill:#667085}} .flow-card{{fill:#fff;stroke:#d9e1ea;stroke-width:1.2;filter:url(#shadow)}}
    .num{{font-size:14px;font-weight:800;fill:#0e5e9f}} .card-title{{font-size:16px;font-weight:800;fill:#12344d}} .card-text{{font-size:12px;fill:#526071}}
    .arrow{{fill:none;stroke:#667085;stroke-width:1.8;marker-end:url(#arrowHead)}} .section{{font-size:18px;font-weight:800;fill:#0f172a}}
  </style>
</defs>
<rect width="1200" height="720" fill="#f6f8fb"/>

<rect class="panel" x="32" y="30" width="342" height="660" rx="10"/>
<rect x="32" y="30" width="342" height="44" fill="#fff" stroke="#e3e8ef"/>
<text class="chat-title" x="52" y="58">{assistant}</text>
<text class="clear" x="330" y="58">{clear}</text>
<rect x="132" y="98" width="190" height="46" rx="8" fill="#f0f7ff" stroke="#a9d0f2"/>
<text class="request" x="170" y="127">{request}</text>
<rect x="54" y="172" width="294" height="504" rx="8" fill="#fff" stroke="#d9e1ea"/>
<text class="assistant" x="70" y="202">{mapgen}</text>
<rect x="70" y="220" width="258" height="38" rx="7" fill="#eaf8ef" stroke="#cbeed8"/>
<circle cx="90" cy="239" r="5" fill="#079669"/>
<text class="done" x="106" y="244">{done}</text>
<text class="reason-title" x="70" y="288">{cards[1][1]}</text>
{"".join(progress_rows)}
<text class="reason-title" x="70" y="426">{cards[3][1]}</text>
{"".join(tool_rows)}
<rect x="54" y="{summary_y}" width="294" height="60" rx="7" fill="#eef8ff" stroke="#a9d8f5"/>
<text class="summary-title" x="70" y="{summary_y + 20}">{summary_title}</text>
{tspan_lines(summary_lines, 70, summary_y + 38, "summary", 16)}

{"".join(card_svg)}
{arrows}
<rect x="430" y="540" width="690" height="92" rx="8" fill="#fff" stroke="#d9e1ea" filter="url(#shadow)"/>
<text class="card-title" x="460" y="574">{'流程含义' if zh else 'Workflow Meaning'}</text>
<text class="card-text" x="460" y="604">{'智能体维护任务状态，依据数据评估选择工具，并通过质量复核形成闭环。' if zh else 'The agent keeps task state, selects tools from data review, and closes the loop with quality checks.'}</text>
</svg>"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for lang in ("zh", "en"):
        svg = flow_svg(lang)
        name = f"ai_dialog_flow_{lang}.svg"
        out = OUT_DIR / name
        out.write_text(svg, encoding="utf-8")
        (PAPER_DIR / name).write_text(svg, encoding="utf-8")
        print(out)
    blank = blank_template_svg()
    for base in (OUT_DIR, PAPER_DIR):
        out = base / "ai_dialog_flow_blank_template.svg"
        out.write_text(blank, encoding="utf-8")
        print(out)


def blank_template_svg() -> str:
    card_svg = []
    positions = [(430, 74), (670, 74), (910, 74), (430, 294), (670, 294), (910, 294)]
    for i, (x, y) in enumerate(positions, 1):
        card_svg.append(f"""
<rect class="flow-card" x="{x}" y="{y}" width="210" height="132" rx="8"/>
<circle cx="{x + 30}" cy="{y + 34}" r="17" fill="#e8f3ff" stroke="#a9d0f2"/>
<text class="num" x="{x + 25}" y="{y + 40}">{i}</text>
<line class="blank-line" x1="{x + 66}" y1="{y + 34}" x2="{x + 174}" y2="{y + 34}"/>
<line class="blank-line thin" x1="{x + 58}" y1="{y + 76}" x2="{x + 176}" y2="{y + 76}"/>
<line class="blank-line thin" x1="{x + 58}" y1="{y + 102}" x2="{x + 160}" y2="{y + 102}"/>
""")
    tool_rows = []
    for i in range(4):
        y = 452 + i * 40
        tool_rows.append(f"""
<rect x="54" y="{y}" width="294" height="31" rx="7" fill="#fff" stroke="#d9e1ea"/>
<circle cx="74" cy="{y + 16}" r="5" fill="#079669"/>
<line class="blank-line tiny" x1="94" y1="{y + 17}" x2="210" y2="{y + 17}"/>
<rect x="242" y="{y + 6}" width="88" height="19" rx="10" fill="#c9f7df"/>
""")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720">
<defs>
  <marker id="arrowHead" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="8" markerHeight="8" orient="auto">
    <path d="M0 0 L10 5 L0 10z" fill="#667085"/>
  </marker>
  <filter id="shadow" x="-8%" y="-8%" width="116%" height="124%">
    <feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#101828" flood-opacity=".12"/>
  </filter>
  <style>
    text{{font-family:"Microsoft YaHei","PingFang SC","Segoe UI",Arial,sans-serif;letter-spacing:0;fill:#17202a}}
    .panel{{fill:#fff;stroke:#d9e1ea;stroke-width:1.2;filter:url(#shadow)}}
    .flow-card{{fill:#fff;stroke:#d9e1ea;stroke-width:1.2;filter:url(#shadow)}}
    .arrow{{fill:none;stroke:#667085;stroke-width:1.8;marker-end:url(#arrowHead)}}
    .num{{font-size:14px;font-weight:800;fill:#0e5e9f}}
    .blank-line{{stroke:#c8d3df;stroke-width:5;stroke-linecap:round}}
    .blank-line.thin{{stroke-width:3.5}}
    .blank-line.tiny{{stroke-width:3}}
  </style>
</defs>
<rect width="1200" height="720" fill="#f6f8fb"/>
<rect class="panel" x="32" y="30" width="342" height="660" rx="10"/>
<rect x="32" y="30" width="342" height="44" fill="#fff" stroke="#e3e8ef"/>
<line class="blank-line" x1="54" y1="54" x2="154" y2="54"/>
<line class="blank-line thin" x1="316" y1="54" x2="354" y2="54"/>
<rect x="132" y="98" width="190" height="46" rx="8" fill="#f0f7ff" stroke="#a9d0f2"/>
<line class="blank-line thin" x1="166" y1="121" x2="288" y2="121"/>
<rect x="54" y="172" width="294" height="504" rx="8" fill="#fff" stroke="#d9e1ea"/>
<line class="blank-line" x1="70" y1="202" x2="158" y2="202"/>
<rect x="70" y="220" width="258" height="38" rx="7" fill="#eaf8ef" stroke="#cbeed8"/>
<circle cx="90" cy="239" r="5" fill="#079669"/>
<line class="blank-line thin" x1="106" y1="239" x2="164" y2="239"/>
<line class="blank-line thin" x1="70" y1="288" x2="142" y2="288"/>
<line class="blank-line tiny" x1="70" y1="320" x2="135" y2="320"/><line class="blank-line tiny" x1="162" y1="320" x2="306" y2="320"/>
<line class="blank-line tiny" x1="70" y1="346" x2="135" y2="346"/><line class="blank-line tiny" x1="162" y1="346" x2="306" y2="346"/>
<line class="blank-line tiny" x1="70" y1="372" x2="135" y2="372"/><line class="blank-line tiny" x1="162" y1="372" x2="238" y2="372"/>
<line class="blank-line tiny" x1="70" y1="398" x2="135" y2="398"/><line class="blank-line tiny" x1="162" y1="398" x2="286" y2="398"/>
<line class="blank-line thin" x1="70" y1="426" x2="150" y2="426"/>
{"".join(tool_rows)}
<rect x="54" y="612" width="294" height="60" rx="7" fill="#eef8ff" stroke="#a9d8f5"/>
<line class="blank-line thin" x1="70" y1="632" x2="170" y2="632"/>
<line class="blank-line tiny" x1="70" y1="650" x2="302" y2="650"/>
<line class="blank-line tiny" x1="70" y1="666" x2="244" y2="666"/>
{"".join(card_svg)}
<path class="arrow" d="M640 140 H660"/>
<path class="arrow" d="M880 140 H900"/>
<path class="arrow" d="M1015 206 V258 C1015 282 998 294 980 294"/>
<path class="arrow" d="M910 360 H890"/>
<path class="arrow" d="M670 360 H650"/>
<rect x="430" y="540" width="690" height="92" rx="8" fill="#fff" stroke="#d9e1ea" filter="url(#shadow)"/>
<line class="blank-line" x1="460" y1="574" x2="590" y2="574"/>
<line class="blank-line thin" x1="460" y1="604" x2="1050" y2="604"/>
</svg>"""


if __name__ == "__main__":
    main()
