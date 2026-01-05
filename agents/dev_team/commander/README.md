# AI Commander

AI Commander 是 dev_team 的默认协作编排器，内置能力探测、共识机制与交叉核查，采用自动模式选择（默认最优 / 局部最优），零配置运行。

## ✅ 快速开始

```python
from agents.dev_team.commander import Commander
from agents.dev_team.utils import load_config

config = load_config()
commander = Commander(config)
commander.initialize_team(planner_result)
result = commander.run_collaboration(max_rounds=5)
```

## 🔧 配置说明

`agents/dev_team/config/config.yaml` 仅保留角色提示词与报告开关，其他为可选配置：

```yaml
roles:
  engineer:
    prompt_path: "agents/dev_team/prompts/engineer.txt"
  qa:
    prompt_path: "agents/dev_team/prompts/qa.txt"

report:
  enabled: true

clean_output_on_start: true  # 启动时安全清理输出目录（检测到可恢复状态则跳过）

review:
  use_llm: false  # 可选：禁用 LLM 生成审查文档（仅用模板兜底）

context:
  skip_dirs: [".git", ".venv", "__pycache__", ".pytest_cache", "output", "data"]
  deny_globs: ["**/.env", "**/*secret*", "**/*token*", "**/*password*", "**/secrets/**"]

quality_gates:
  require_tests: true
  require_ui_tests: true
  require_coverage: true
  require_ui_simulation: true

testing:
  coverage_command: "pytest --cov=. --cov-report=term-missing"
  ui_test_patterns:
    - "tests/ui/**/*.py"
    - "ui/tests/**/*.py"

ui_design:
  enabled: true
  required: true
  force_if_no_frontend: true
  allow_without_baseline: true
  baseline_path: ""
  use_internal: true
  model: "gemini-2.5-flash-image"
  baseline_name: "design_baseline.png"
  summary_enabled: true
  summary_required: true
  comparison_required: false
  pixel_diff_threshold: 0.15
  layout_similarity_threshold: 0.75
  layout_compare_size: [192, 192]
  edge_threshold: 20

环境变量:
- `GOOGLE_API_KEY` 或 `GEMINI_API_KEY`
- `NANOBANNA_MODEL` (默认 gemini-2.5-flash-image)
- `UI_DESIGN_BASELINE` (可选，用户参考图路径)

外部命令模式（可选）:
- 配置 `use_internal: false` 并提供 `command`, `prompt_arg`, `output_arg`
```

## 📊 工作流程

```
1. 团队初始化
   ├─ 基础协作编排
   └─ 能力探测

2. 审查层产出
   ├─ 头脑风暴记录
   ├─ 设计审查清单
   ├─ 验收清单
   └─ ADR + 现状摘要

3. 多轮迭代
   ├─ 阶段1: Agent工作
   ├─ 阶段2: 共识达成（自动模式）
   ├─ 阶段3: 自动化测试
   ├─ 阶段4: 交叉核查（自动模式）
   ├─ 阶段5: QA审查
   ├─ 证据门禁: UI基线/用户模拟/Evidence Pack
   └─ 交付审批: 关键节点需 APPROVED

4. 结果输出
   ├─ 协作报告
   └─ 能力档案
```

## 🧠 自动模式

- 简单任务：局部最优（跳过共识与交叉核查）
- 中/复杂任务：默认最优（开启共识与交叉核查）
