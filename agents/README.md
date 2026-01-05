# Agents 目录定位说明

本目录存放可对外发布的独立软件/Agent。项目的“母体”是 `agents/dev_team`（默认协作引擎/Commander）：它基于用户需求在其工作区内规划、生成并发布新的软件或 Agent。当前可用的两个对外发布项均由 `dev_team` 工作区产出并发布。

## 母体（dev_team）与产出流程

- 母体职责：`agents/dev_team` 负责理解需求、进行架构与实现决策，自动选择模式并驱动单元测试与验证，最终在其工作区产出对外发布的子项目。
- 产出形式：产出项以独立目录的形式存在于 `agents/<agent_name>/`，并具备自己的 `agent.py`、`prompts/` 与 `config/`。
- 发布入口：可通过 `ui/streamlit_app.py` 聚合并运行发布的 Agent，或直接使用各子项目的脚本/入口。
- 源头标注：每个发布的 Agent 在文档中标注来源为“由 dev_team 工作区产出并发布”。

## 已发布的 Agents

| Agent 名称 | 简述 | 状态 | 来源 |
| :--- | :--- | :--- | :--- |
| **excel_to_csv** | 将 Excel 转换为 CSV 的工具型 Agent | Active | 由 `dev_team` 工作区产出并发布 |
| **restaurant_recommender** | 基于位置、价格与历史记录的餐厅推荐 Agent | Active | 由 `dev_team` 工作区产出并发布 |
| **dev_team** | 母体（默认协作引擎/Commander），负责派生与发布 | Active | 母体（不对外作为单一功能发布项） |

## 如何创建新的 Agent

优先建议通过母体 `dev_team` 的工作流产出新的 Agent（在其工作区内完成需求分析、架构设计、代码生成与测试，并在通过验证后发布到 `agents/<new_agent_name>/`）。

如需手动创建（非推荐，仅在特殊场景使用），请遵循以下规范：
1. 新建目录：`agents/<new_agent_name>/`
2. 添加基础结构：`agent.py`、`prompts/`、`config/`
3. 在 `agent.py` 中实现 `build_agent()` 入口
4. 在本文件与子目录 `README.md` 中补充文档，并明确标注该 Agent 的来源（若非由 `dev_team` 产出则说明理由）
