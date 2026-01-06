结论：dev_team 使用 Commander 作为唯一入口，内置“默认最优/局部最优”的自动模式，零配置运行。

入口
- 启动入口：`agents/dev_team/main.py`

默认行为
- 自动选择模式：
  - 复杂任务 → 默认最优（共识 + 交叉核查）
  - 简单任务 → 局部最优（跳过共识与交叉核查）
- 运行前生成/检查审查层产出（头脑风暴记录、设计审查清单、验收清单、ADR、现状摘要），缺任一阻断进入实现
- 若涉及 UI/前端或项目缺少前端模块，自动调用 nanobanna 生成设计基线并由 AI 读图生成设计摘要；若用户提供参考图或设计师不可用则跳过生成
- summary-only 输出，避免上下文膨胀
- 角色视图过滤上下文：QA 只接收工程输出摘要，工程角色仅接收 QA 反馈摘要，避免重复注入
- 测试自动执行，按单测文件逐个运行
- UI 测试与覆盖率统计自动执行；若要求前端交付则缺失即阻断
- 若存在输入契约测试脚本，自动执行并纳入门禁
- 测试通过后执行用户模拟脚本（如存在），失败阻断交付
- 测试通过后进入交付审批门禁（内置，无需配置）
- 前端任务要求提供 UI 设计基线与实现截图，否则阻断交付
- 前端任务要求设计摘要（AI 读图）与 UI 测试覆盖率，否则阻断交付（允许在无基线时降级；像素级对齐可配置为非阻断）
- 交叉核查的相似度比较对输出进行长度截断，降低性能开销
- 写入代码时会跳过疑似占位/摘要代码（如大量 `...`），防止覆盖可用实现
- 更新模式支持函数级替换：`<file><function name="...">...</function></file>`，用于只替换目标函数体
- 迭代目标拷贝会跳过敏感配置（如 `llm.yaml`、`langsmith.yaml`、`*.local.yaml`），避免将密钥带入输出目录
- 自动生成证据清单 `output/evidence/manifest.json` 并归档为 `output/evidence/evidence_pack_*.zip`
- 门禁失败自动生成 Bug Card 并记录至 `output/evidence/bugs/`
- UI 截图对齐元数据生成 `output/evidence/ui/comparison.json`
- UI 像素级差异图生成 `output/evidence/ui/diff.png`（超阈值阻断交付）
 - 迭代模式：传入 `iteration_target` 即为 1→100；留空为 0→1

配置
- `agents/dev_team/config/config.yaml` 提供默认 max_rounds=5、UI 设计/测试/覆盖率门禁配置
- LLM 密钥可通过环境变量 `LLM_API_KEY` 或 `DEEPSEEK_API_KEY` 提供
