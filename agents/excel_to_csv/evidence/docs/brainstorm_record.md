# 头脑风暴记录
- 任务: 对现有 excel_to_csv 模块进行重构与优化，在保持功能一致性的前提下，重点提升性能、改善UI体验、实现模块化架构，并满足严格的验收标准。
- 日期: 2026-01-05
- 输入: AST 基线 /home/minghan/project/agents/agents/dev_team/output/codebase/evidence/ast/ast_baseline.md；现状摘要 未提供

## 约束
- 目标: 提升性能（skip_llm=false < 90s, skip_llm=true < 30s），改善UI，实现模块化，保持功能一致性。
- 边界: 基于现有项目路径 /home/minghan/project/agents/agents/excel_to_csv 进行开发，不破坏现有CLI接口。
- 非功能需求: 并发安全（默认上限），UI模块化布局（基础/高级/审计），配置持久化，代码质量（测试覆盖率>=85%）。
- 输入契约: 现有CLI参数及行为必须完整支持。

## 方案A：渐进式重构与性能优化
- 描述: 1) 首先分析现有代码性能瓶颈（如长函数 `_process_dataframe`, `_ai_review_and_transform`）。2) 将核心逻辑（文件扫描、工作表处理、AI审核、CSV写入）抽取为独立模块（core/）。3) 引入 `concurrent.futures.ThreadPoolExecutor` 实现文件级和表级并发，设置默认上限为 `min(32, os.cpu_count() * 2)`。4) 使用 `streamlit` 构建三模块UI，通过回调调用解耦后的核心模块。5) 性能优化：对Pandas操作进行向量化，缓存AI模型加载，优化I/O（批量写入）。6) 保持现有 `agent.py` 的CLI入口，内部调用新模块。
- 质量/稳定性/可读性/工程性/模块化: 高。通过模块化提升可读性和可维护性。渐进式重构降低风险。明确的并发上限和资源管理提升稳定性。
- 风险与缓解: 风险：重构可能引入功能不一致。缓解：编写详尽的集成测试，对比重构前后输出文件的哈希值。风险：性能优化可能复杂。缓解：使用 profiling 工具（如 cProfile）定位瓶颈，优先优化热点。

## 方案B：激进式重写与架构革新
- 描述: 1) 完全重写核心逻辑，采用异步架构（asyncio + aiofiles）。2) 设计全新的插件化架构，将文件解析、转换、审核等步骤定义为可插拔的管道。3) 构建高级Streamlit UI，支持实时可视化管道执行。4) 通过适配器模式兼容现有CLI接口。
- 质量/稳定性/可读性/工程性/模块化: 极高模块化和工程性，但稳定性风险高。异步架构可能带来复杂性和调试难度。
- 风险与缓解: 风险：开发周期长，与现有功能一致性难以保证。缓解：需要并行运行新旧两套系统进行对比验证，增加测试负担。风险：团队可能缺乏异步编程经验。缓解：提供详细的设计文档和培训。

## 备选方案C：最小化修改，仅优化性能与UI
- 描述: 1) 仅对现有 `agent.py` 中的长函数进行内部重构（提取子函数）。2) 在现有代码中直接添加并发控制（修改 `convert_excel_dir`）。3) 单独开发一个Streamlit UI包装器，通过子进程调用现有CLI。
- 质量/稳定性/可读性/工程性/模块化: 低。可读性和模块化改善有限，但改动最小，风险最低。工程性差，UI与核心逻辑耦合。
- 风险与缓解: 风险：架构未解耦，长期维护成本高。缓解：作为短期方案，后续仍需彻底重构。风险：性能提升可能有限。缓解：聚焦于最耗时的操作（如AI调用、大文件I/O）进行优化。

## 选择理由
- 结论: 选择方案A（渐进式重构与性能优化）。它在风险可控的前提下，较好地平衡了架构改进、性能提升和UI体验的目标。符合“基于现有代码修改”的更新规则，能够逐步交付价值，并确保功能一致性。

## 拆解建议
- 触发条件: 函数超过 50 行
- 建议清单:
  1.  `agent.py:_process_dataframe` (130行): 拆分为 `_detect_header`, `_clean_dataframe`, `_transform_if_needed`, `_validate_output`。
  2.  `agent.py:_ai_review_and_transform` (91行): 拆分为 `_call_llm_for_review`, `_parse_feedback`, `_execute_transformations`。
  3.  `agent.py:convert_excel_dir` (87行): 拆分为 `_scan_input_files`, `_schedule_concurrent_tasks`, `_aggregate_results`。将并发逻辑移入新模块 `concurrency_manager.py`。
  4.  `agent.py:_build_long_format` (59行): 可保持，但检查内部逻辑是否可提取 helper 函数。
  5.  `agent.py:_apply_feedback` (53行): 拆分为 `_apply_column_rename`, `_apply_transpose`, `_apply_cleanup`。