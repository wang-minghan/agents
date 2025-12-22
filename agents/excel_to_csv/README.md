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
```

## Poetry 使用

```bash
poetry install
poetry run python agents/excel_to_csv/agent.py --input <input_dir> --output <output_dir>
```

## LLM 配置

- 默认读取 `configs/llm.yaml`
- 可通过 `--llm-config` 指定

示例字段（多模型可选）：

```yaml
active_profile: grok
profiles:
  grok:
    provider: xai
    api_base: "https://api.x.ai/v1"
    model: "grok-4.1-fast"
    model_reasoner: "grok-4.1-fast"
    api_key: "<your_key_here>"
  deepseek:
    provider: deepseek
    api_base: "https://api.deepseek.com"
    model: "deepseek-chat"
    model_reasoner: "deepseek-reasoner"
    api_key: "<your_key_here>"
common:
  max_workers: 2
  timeout_seconds: 120
  max_retries: 2
```

模型选择可配置（可选）：

```yaml
reasoner_min_cols: 80
reasoner_min_sheet_name_len: 12
force_reasoner: false
force_chat: false
```

## LangSmith 配置

- 默认读取 `configs/langsmith.yaml`
- 可通过 `--langsmith-config` 指定

示例字段：

```yaml
enabled: true
api_key: "<your_key_here>"
endpoint: "https://api.smith.langchain.com"
project: "excel_to_csv"
```

## 配置

- `agents/excel_to_csv/config/config.yaml`
