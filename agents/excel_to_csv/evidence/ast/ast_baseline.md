root=/home/minghan/project/agents/agents/dev_team/output/codebase

[README.md]
# Excel to CSV Agent

将多页面（多工作表）Excel 批量转换为 CSV，保持目录结构尽量不变，输出更易读的 CSV 文件。

## 行为

- 输入目录递归扫描 `.xls/.xlsx/.xlsm/.csv`
- 输出目录保留原有层级
- 多工作表输出为 `<主题_表名_版本>.csv`
- 生成 `*_audit.md` 结构审核报告
- 允许在不破坏信息的前提下进行列名语义化、转置或清理空行空列
- 最终审核员最多迭代 5 轮，不通过则保持现状并输出问题清单
- 支持多行表头合并（由 AI 识别表头行）
- 多工作表并行处理（线程池）
- 默认允许转长表（可在配置中关闭）

## 使用

```bash
python agents/excel_to_csv/agent.py --input <input_dir> --output <output_dir>
``

[config/config.yaml]
# Excel to CSV agent config
input_dir: "<input_dir>"
output_dir: "<output_dir>"


[evidence/ast/ast_baseline.md]
root=/home/minghan/project/agents/agents/dev_team/output/codebase

[README.md]
# Excel to CSV Agent

将多页面（多工作表）Excel 批量转换为 CSV，保持目录结构尽量不变，输出更易读的 CSV 文件。

## 行为

- 输入目录递归扫描 `.xls/.xlsx/.xlsm/.csv`
- 输出目录保留原有层级
- 多工作表输出为 `<主题_表名_版本>.csv`
- 生成 `*_audit.md` 结构审核报告
- 允许在不破坏信息的前提下进行列名语义化、转置或清理空行空列
- 最终审核员最多迭代 5 轮，不通过则保持现状并输出问题清单
- 支持多行表头合并（由 AI 识别表头行）
- 多工作表并行处理（线程池）
- 默认允许转长表（可在配置中关闭）

## 使用

```bash
p

[evidence/docs/acceptance_checklist.md]
# 验收清单
- 任务: excel_to_csv 模块重构与优化
- 日期: 2026-01-05

## 验收项
- [ ] **功能一致性**：重构后模块处理相同输入的输出（CSV文件内容、审计报告）必须与现有模块完全一致（通过自动化集成测试验证）。
- [ ] **性能达标**：使用标准测试集（单个Excel，10 sheets, 5万行/sheet, 20列），端到端耗时满足：skip_llm=false模式 < 90秒，skip_llm=true模式 < 30秒（起止点：从CLI命令执行开始到所有文件写入完成）。
- [ ] **并发安全**：并发处理功能正常工作，并实现了可配置的默认安全上限（如最大工作线程数不超过CPU核心数的2倍），无资源泄漏或竞争条件。
- [ ] **架构解耦**：成功实现UI层、核心业务逻辑层、配置管理层的清晰分离，模块间通过定义良好的接口通信。
-

[evidence/docs/adr.md]
# 架构决策记录 (ADR)

## ADR 001: Excel to CSV 模块重构架构

### 状态
已接受

### 背景
现有 `excel_to_csv` 模块存在以下问题：
1. 代码耦合度高：UI逻辑、业务逻辑、配置管理混在一起
2. 性能瓶颈：并发控制简单，无法充分利用多核CPU
3. 缺乏配置持久化：运行时设置无法保存和复用
4. 测试覆盖不足：缺乏单元测试和性能测试

### 决策
采用分层架构设计，将系统拆分为以下核心模块：

#### 1. 配置管理模块 (`config_manager.py`)
- 负责加载、保存、验证配置
- 支持 JSON/YAML 格式
- 提供默认配置和安全范围验证

#### 2. 核心数据处理模块 (`data_processor.py`)
- 纯业务逻辑，无UI依赖
- 实现高效的Excel/CSV读写
- 支持表级和文件级并

[evidence/docs/brainstorm_record.md]
# 头脑风暴记录
- 任务: 对现有 excel_to_csv 模块进行重构与优化，在保持功能一致性的前提下，重点提升性能、改善UI体验、实现模块化架构，并满足严格的验收标准。
- 日期: 2026-01-05
- 输入: AST 基线 /home/minghan/project/agents/agents/dev_team/output/codebase/evidence/ast/ast_baseline.md；现状摘要 未提供

## 约束
- 目标: 提升性能（skip_llm=false < 90s, skip_llm=true < 30s），改善UI，实现模块化，保持功能一致性。
- 边界: 基于现有项目路径 /home/minghan/project/agents/agents/excel_to_csv 进行开发，不破坏现有CLI接口。
- 非功能需求: 并发安全（默认上限），

[evidence/docs/current_state_summary.md]
# 现状摘要
- 任务: 对现有 excel_to_csv 模块进行重构与优化，在保持功能一致性的前提下，重点提升性能、改善UI体验、实现模块化架构，并满足严格的验收标准。
- 日期: 2026-01-05
- AST 基线路径: /home/minghan/project/agents/agents/dev_team/output/codebase/evidence/ast/ast_baseline.md

## 功能概览
未提供

## 关键路径
未提供

## 已知风险
未提供

## 长函数热点
- agent.py: _process_dataframe (130 行)
- agent.py: _ai_review_and_transform (91 行)
- agent.py: convert_excel_dir (87 行)
- agent.py: _build_long_f

[evidence/docs/design_review_checklist.md]
# 设计审查清单
- 任务: 对现有 excel_to_csv 模块进行重构与优化
- 日期: 2026-01-05

## 质量
- [x] 验收清单齐全 (见 acceptance_checklist.md)
- [x] 证据要求明确 (需求中明确了审查产物、ADR、证据链等交付物)
- [x] 关键路径覆盖 (头脑风暴中分析了核心处理路径和并发路径)

## 稳定性
- [x] 不可变约束明确（目标/边界/输入契约） (约束中明确了功能一致性、性能约束、兼容性约束)
- [x] 关键阈值定义清楚（性能/可靠性） (性能约束: skip_llm=false < 90s, skip_llm=true < 30s；并发安全上限需定义)
- [x] 失败策略明确（不允许 silent fail） (需在设计中加入异常处理、日志记录和UI反馈)

## 可读性
- [x] 关键路径可解释 (方

[evidence/docs/evidence/docs/adr.md]
# 架构决策记录 (ADR)

## ADR 001: Excel to CSV 模块重构架构

### 状态
已接受

### 背景
现有 `excel_to_csv` 模块存在以下问题：
1. 代码耦合度高：UI逻辑、业务逻辑、配置管理混在一起
2. 性能瓶颈：并发控制简单，无法充分利用多核CPU
3. 缺乏配置持久化：运行时设置无法保存和复用
4. 测试覆盖不足：缺乏单元测试和性能测试

### 决策
采用分层架构设计，将系统拆分为以下核心模块：

#### 1. 配置管理模块 (`config_manager.py`)
- 负责加载、保存、验证配置
- 支持 JSON/YAML 格式
- 提供默认配置和安全范围验证

#### 2. 核心数据处理模块 (`data_processor.py`)
- 纯业务逻辑，无UI依赖
- 实现高效的Excel/CSV读写
- 支持表级和文件级并

[prompts/system.md]
你是一个用于 Excel 转 CSV 的智能体。
目标：将多工作表 Excel 转换为可读性更强的 CSV，保持目录结构尽量不变，内容不改写。
输出：按原目录结构写入 output 目录，文件命名为 <原文件名>__<工作表名>.csv。


[tests/README.md]
# Tests

可在此补充 Excel 转换的单测与回归样例。
