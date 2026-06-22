"""
技能管理模块
允许 Agent 保存和复用编写的脚本（学习能力）
"""
import os
import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


BUILTIN_SKILLS: Dict[str, Dict[str, str]] = {
    "map-generalization": {
        "name": "map-generalization",
        "description": "地图综合工作流：按源/目标比例尺进行算子选择、冲突检测、迭代优化与质量验收；支持点分布结构自适应（corridor/clustered/uniform）。",
        "usage_example": (
            "任务示例：将当前道路图层从 1:10000 综合到 1:50000。\n"
            "执行要点：先识别几何类型与冲突密度，再按 selection -> simplify/typification -> smooth/merge -> displacement 迭代，最后输出文件与质量报告。"
        ),
        "code": (
            "# skill: map-generalization\n"
            "# 1) 定义 source_scale/target_scale 与可读性阈值\n"
            "# 2) 按几何类型和点结构(corridor/clustered/uniform)选择算子并设置参数\n"
            "# 3) 迭代执行 + 质量校验（拓扑、冲突、保留率）\n"
            "# 4) 运行 VLM 验收（视觉结构保持）\n"
            "# 5) 输出 generalized dataset + 参数表 + 质量说明\n"
        ),
    }
}


class SkillManager:
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = skills_dir
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir)

    def save_skill(self, name: str, description: str, code: str, usage_example: str = "") -> Dict:
        """保存技能"""
        try:
            skill_data = {
                "name": name,
                "description": description,
                "code": code,
                "usage_example": usage_example,
            }

            file_path = os.path.join(self.skills_dir, f"{name}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(skill_data, f, ensure_ascii=False, indent=2)

            return {"status": "success", "message": f"技能 '{name}' 已保存"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_skill(self, name: str) -> Optional[Dict]:
        """获取技能"""
        builtin = BUILTIN_SKILLS.get(name)
        if builtin:
            return dict(builtin)

        md_path = os.path.join(self.skills_dir, f"{name}.md")
        if os.path.exists(md_path):
            try:
                with open(md_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                return {
                    "name": name,
                    "description": self._extract_md_title(content) or f"Markdown skill: {name}",
                    "code": content,
                    "usage_example": "Loaded from local markdown skill file.",
                }
            except Exception as e:
                logger.error(f"Error loading markdown skill {name}: {e}")

        file_path = os.path.join(self.skills_dir, f"{name}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading skill {name}: {e}")
        return None

    def list_skills(self) -> List[Dict]:
        """列出所有技能"""
        skills = [
            {"name": data.get("name", name), "description": data.get("description", "")}
            for name, data in BUILTIN_SKILLS.items()
        ]

        if not os.path.exists(self.skills_dir):
            return skills

        for filename in os.listdir(self.skills_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.skills_dir, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        skills.append({
                            "name": data.get("name"),
                            "description": data.get("description"),
                        })
                except Exception:
                    continue
            elif filename.endswith(".md"):
                try:
                    path = os.path.join(self.skills_dir, filename)
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    name = os.path.splitext(filename)[0]
                    skills.append(
                        {
                            "name": name,
                            "description": self._extract_md_title(content) or f"Markdown skill: {name}",
                        }
                    )
                except Exception:
                    continue
        return skills

    def _extract_md_title(self, content: str) -> str:
        for line in (content or "").splitlines():
            t = line.strip()
            if t.startswith("#"):
                return t.lstrip("#").strip()
        return ""


# 全局实例
skill_manager = SkillManager()
