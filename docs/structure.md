# 项目结构与规范

本项目使用 LangChain 搭建多 Agent 体系。以下结构旨在支持多团队并行开发、复用共享组件，并保持清晰的边界。

## 推荐目录结构

```
.
├─ agents/            # 每个 Agent 独立目录
│  ├─ <agent_name>/
│  │  ├─ agent.py     # 入口/构建函数
│  │  ├─ prompts/     # 提示词与模板
│  │  ├─ tools/       # 仅此 Agent 使用的工具
│  │  ├─ chains/      # 仅此 Agent 使用的链
│  │  ├─ memory/      # 仅此 Agent 使用的记忆实现
│  │  ├─ config/      # Agent 专属配置
│  │  └─ tests/       # Agent 局部测试
│  └─ README.md       # 所有 Agent 清单与说明
├─ shared/            # 通用模块（工具、链、prompt、memory）
│  ├─ prompts/
│  ├─ tools/
│  ├─ chains/
│  ├─ memory/
│  └─ utils/
├─ workflows/         # 多 Agent 编排流程与运行图
├─ tools/             # 全局工具（所有 Agent 可用）
├─ configs/           # 全局配置（环境、模型、路由等）
├─ scripts/           # 运维/数据/批处理脚本
├─ tests/             # 集成测试与端到端测试
├─ data/              # 本地样例数据（如需）
├─ docs/              # 设计、规范与开发文档
└─ ui/                # 简单 UI（例如 Streamlit）
```

## 命名与边界

- 目录命名：全小写，使用下划线分隔（如 `customer_support`）。
- `agents/<agent_name>` 内只放该 Agent 专属内容；可复用的逻辑必须上移到 `shared/` 或 `tools/`。
- `tools/` 内的工具要求接口稳定、可复用，并配套文档与测试。
- `workflows/` 只负责编排，不承载业务实现。

## Agent 规范

- 每个 Agent 至少包含：`agent.py`、`prompts/`、`config/`。
- `agent.py` 建议提供工厂方法：`build_agent(config)`。
- 统一在 `config/` 中声明可配置参数，避免在代码中硬编码模型名或 API Key。

## 配置规范

- 全局配置放在 `configs/`，例如 `configs/base.yaml`、`configs/dev.yaml`。
- Agent 专属配置放在 `agents/<agent_name>/config/`。
- 机密信息不入库，使用环境变量或本地 `.env` 文件加载。

## 共享组件规范

- 共享模块需有最小可用示例和单测。
- `shared/prompts/` 与 `shared/tools/` 内的文件要求有简短说明，保持可搜索性。

## 测试规范

- Agent 内部测试放在 `agents/<agent_name>/tests/`。
- 跨 Agent 或流程级测试放在 `tests/`。

## 新增 Agent 步骤

1. 在 `agents/` 创建目录 `agents/<agent_name>/`。
2. 编写 `agent.py` 与 `prompts/`。
3. 若有可复用逻辑，先放入 `shared/` 再引用。
4. 在 `agents/README.md` 增加条目。
