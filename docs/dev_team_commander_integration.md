结论：dev_team 使用 Commander 作为唯一入口，内置“默认最优/局部最优”的自动模式，零配置运行。

入口
- 启动入口：`agents/dev_team/main.py`

默认行为
- 自动选择模式：
  - 复杂任务 → 默认最优（共识 + 交叉核查）
  - 简单任务 → 局部最优（跳过共识与交叉核查）
- summary-only 输出，避免上下文膨胀
- 测试自动执行，按单测文件逐个运行
- 测试通过后执行用户模拟脚本（如存在），失败阻断交付
- 测试通过后进入交付审批门禁（内置，无需配置）
- 前端任务要求提供 UI 设计基线与实现截图，否则阻断交付
- 自动生成证据清单 `output/evidence/manifest.json` 并归档为 `output/evidence/evidence_pack_*.zip`
- UI 截图对齐元数据生成 `output/evidence/ui/comparison.json`
- UI 像素级差异图生成 `output/evidence/ui/diff.png`（超阈值阻断交付）
 - 迭代模式：传入 `iteration_target` 即为 1→100；留空为 0→1

配置
- `agents/dev_team/config/config.yaml` 仅保留角色提示词与报告开关
