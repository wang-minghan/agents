文档类型｜目标读者｜核心问题｜读完能做
- 类型：规范/方案
- 读者：项目负责人、dev_team 维护者、QA
- 核心问题：如何发现bug并形成可交付证据链；如何把反馈转化为AI自修复闭环；如何标准化流程与门禁
- 读完能做：建立统一缺陷模型与证据要求；接入自动化回归与自修复流程；定义交付门禁与可控输出

结论
- 采用“证据驱动 + 门禁闭环 + 自修复回合”的统一质量流程：需求基线 → 设计基线 → 实现 → 证据验证 → 反馈 → 自动修复 → 再验证 → 交付。
- 关键是三件事：统一缺陷模型、强制证据产出、自动化回归与修复循环。
- 0→1 与 1→100 只需要切换“基线构建方式与门禁强度”，流程不变。
- UI 设计基线默认使用 nanobanna 生成，并由 AI 读图产出可实现的设计摘要；若设计师不可用或用户已提供参考图，可降级为“仅实现截图 + UI 测试”模式。

零一与一到100场景适配（不改主流程）
- 0→1：先冻结“不可变约束”（目标、边界、非功能需求、输入契约），再允许实现层快速试错；设计基线与接口契约必须先于代码。
- 1→100：以现有系统为基线，先生成 AST/行为/性能基线，再做函数级替换与兼容性校验；禁止结构性重构未附迁移与回滚方案。

一、质量闭环标准（流程）
- 需求确认：冻结基线与验收清单（Acceptance Checklist）
- 设计基线：UI结构/尺寸/交互/状态基线（Design Baseline）
- 实现交付：代码产物+变更摘要
- 证据验证：截图/测试报告/输入鲁棒性结果
- 缺陷归档：统一缺陷模型（Bug Card）
- 自动修复：AI依据 Bug Card 修复并提交
- 再验证：复测通过后进入交付
- 交付门禁：必须“验收清单+证据+复测通过”三项齐全

二、缺陷模型（Bug Card）
必填字段（最小可用）：
- id：唯一ID
- type：UI/逻辑/输入/性能/安全/数据一致性
- severity：S1-S4
- scope：模块/页面/组件
- repro_steps：复现步骤
- expected：期望结果
- actual：实际结果
- evidence：截图/日志/测试输出
- owner：修复责任人（AI角色）
- status：open/fixed/verified

三、证据标准（Evidence Pack）
- UI问题：nanobanna 设计基线图 + 读图设计摘要 + 当前实现截图 + 对齐说明（允许“布局相似即可”，采用布局相似度度量）
- 证据存放规范：`output/evidence/ui/design_baseline.*` 与 `output/evidence/ui/implementation.*`
- 读图摘要存放：`output/evidence/docs/ui_design_summary.md`
- 自动对齐检查：生成 `output/evidence/ui/comparison.json`（hash/尺寸对齐证据）
- 像素级对比：生成 `output/evidence/ui/diff.png`，阈值与是否阻断可配置；布局相似度可独立配置阈值
- 输入问题：输入契约 + 边界样例 + 失败日志
- 逻辑问题：最小复现用例 + 单测输出
- 交付要求：Evidence Pack 与 Bug Card 绑定归档
- 用户模拟：覆盖路径清单 + 关键截图 + 自动化输出
- 归档要求：生成 `output/evidence/manifest.json` 与 `output/evidence/evidence_pack_*.zip`
- 迭代基线：生成 `output/evidence/ast/ast_baseline.md` 记录现有代码结构
- 断点与基线同步：中断后从 `output/evidence/collaboration_state.json` 续跑；证据与设计文档同步到迭代目标根目录 `evidence/`

四、输入鲁棒性标准
- 输入契约：类型、范围、空值、默认值、容错策略
- 失败策略：统一返回错误与提示，不允许 silent fail
- 单测覆盖：正常、边界、异常、混合类型

五、自动修复流程（AI Self-Fix Loop）
- 触发：检测到 Bug Card open
- 读取：Bug Card + Evidence Pack + 相关代码摘要
- 修复：生成补丁 + 说明 + 新增/更新测试
- 验证：运行单测 + 关键路径回归
- 归档：将修复结果写回 Bug Card

六、交付门禁（Release Gate）
- 通过条件：
  - 验收清单完成
  - Evidence Pack 完整
  - 复测通过
  - 若为前端任务：必须存在设计基线、读图摘要、实现截图与 UI 测试覆盖率
- 未通过：阻断交付，自动进入修复回合

七、稳定性与灵活性控制
- 稳定性：不可变约束清单（目标/边界/输入契约/性能阈值/安全红线），任何修改必须提供变更理由与影响面证据。
- 灵活性：可变策略层（算法/实现/框架选择）允许替换，但必须满足“契约一致 + 证据一致”。
- 防返工：输出前做“头脑风暴 + 设计评审 + 契约评审 + 复杂度评审”，先形成结论再落地实现。
- 重构策略：鼓励重构；当函数超过 50 行时必须给出拆解建议，并记录模块化收益与风险。

八、最小度量（避免伪指标）
- 结构：模块数/依赖层级/循环依赖（有即阻断）
- 风险：复杂度热点（函数/模块 top 5）
- 变更：diff 规模与影响面（触达模块数）
- 质量：关键路径用例通过率 + 失败回归次数

九、审查层与模板（必须产出）
- 产出要求：每次开发必须同时产出“头脑风暴记录 + 设计审查清单 + 验收清单 + 决策记录（ADR）”。
- 阻断条件：缺任一产出即视为未通过设计审核，禁止进入实现。
- 存放位置：`output/evidence/docs/brainstorm_record.md`、`output/evidence/docs/design_review_checklist.md`、`output/evidence/docs/acceptance_checklist.md`、`output/evidence/docs/adr.md`、`output/evidence/docs/current_state_summary.md`

头脑风暴记录（Brainstorm Record）
- 目标：至少产出 2 个可行方案与 1 个备选方案。
- 输入：必须包含 AST 基线与现状摘要（功能、关键路径、已知风险），作为变更讨论的起点。
- 结论字段：
  - 约束：目标/边界/非功能需求/输入契约
  - 方案对比：质量/稳定性/可读性/工程性/模块化
  - 风险：失败模式与缓解
  - 选择理由：为何此时此解

设计审查清单（Design Review Checklist）
- 质量：是否满足验收清单与证据要求
- 稳定性：是否明确不可变约束与阈值
- 可读性：关键路径是否可解释，命名/层次是否一致
- 工程性：是否有迁移、回滚、监控/日志方案
- 模块化：边界是否清晰，耦合是否可控

决策记录（ADR）最小模板
- title：决策标题
- status：proposed/accepted/rejected
- context：背景与约束
- decision：最终选择
- alternatives：备选方案与取舍
- consequences：短期影响 + 长期维护成本

七、默认策略（零配置）
- summary-only 上下文输出
 - 测试默认单次运行（-x 早停）
- 审批仅保留“项目/规划最终发布确认”
- 其余流程自动执行，无需人工选择
- 测试通过后执行用户模拟脚本（如存在），失败即阻断交付
- 若为 1→100 迭代，自动记录 AST 基线并注入规划约束
- 若提供 UI 设计基线/实现截图，需求分析阶段自动走多模态输入

行动清单
- What：将质量闭环接入 dev_team 主流程；Who：我；When：本轮
- What：定义 Bug Card 与 Evidence Pack 模板；Who：我；When：本轮
- What：把 UI 截图/输入鲁棒/单测回归接入门禁；Who：我；When：本轮
