# AI Commander

AI Commander 是 dev_team 的默认协作编排器，内置能力探测、共识机制与交叉核查，采用自动模式选择（默认最优 / 局部最优），零配置运行。

## ✅ 快速开始

```python
from agents.dev_team.commander import Commander
from agents.dev_team.utils import load_config

config = load_config()
commander = Commander(config)
commander.initialize_team(planner_result)
result = commander.run_collaboration(max_rounds=3)
```

## 🔧 配置说明

`agents/dev_team/config/config.yaml` 仅保留角色提示词与报告开关：

```yaml
roles:
  engineer:
    prompt_path: "agents/dev_team/prompts/engineer.txt"
  qa:
    prompt_path: "agents/dev_team/prompts/qa.txt"

report:
  enabled: true
```

## 📊 工作流程

```
1. 团队初始化
   ├─ 基础协作编排
   └─ 能力探测

2. 多轮迭代
   ├─ 阶段1: Agent工作
   ├─ 阶段2: 共识达成（自动模式）
   ├─ 阶段3: 单元测试逐个执行
   ├─ 阶段4: 交叉核查（自动模式）
   └─ 阶段5: QA审查

3. 结果输出
   ├─ 协作报告
   └─ 能力档案
```

## 🧠 自动模式

- 简单任务：局部最优（跳过共识与交叉核查）
- 中/复杂任务：默认最优（开启共识与交叉核查）
