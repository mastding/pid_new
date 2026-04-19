"""数据理解类技能：负责数据加载、清洗、特征识别。"""
# 副作用导入：每个模块通过 @register 自注册
from core.skills.data_understanding import load_dataset  # noqa: F401
from core.skills.data_understanding import detect_candidate_windows  # noqa: F401
from core.skills.data_understanding import summarize_data  # noqa: F401
