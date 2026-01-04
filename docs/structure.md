# 项目结构与规范

本项目以最小学习成本为目标，默认入口与配置足够完成全流程，无需额外开关。

## 当前目录结构

```
.
├─ agents/            # 所有 Agent
│  ├─ dev_team/       # 默认协作引擎（Commander）
│  ├─ dev_team/architect/   # 架构师（规划）
│  ├─ excel_to_csv/   # 样例Agent
│  └─ restaurant_recommender/  # 样例Agent
├─ configs/           # 全局配置
├─ docs/              # 文档
├─ ui/                # Streamlit UI
├─ data/              # 样例数据
├─ scripts/           # 脚本
└─ tests/             # 测试
```

## 约定

- `agents/<agent_name>` 内只放该 Agent 专属内容
- 对外入口优先 `agents/dev_team/main.py` 或 `ui/streamlit_app.py`
- 配置仅保留必须项，默认行为由代码自动选择
