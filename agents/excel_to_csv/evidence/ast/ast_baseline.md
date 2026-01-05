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


[prompts/system.md]
你是一个用于 Excel 转 CSV 的智能体。
目标：将多工作表 Excel 转换为可读性更强的 CSV，保持目录结构尽量不变，内容不改写。
输出：按原目录结构写入 output 目录，文件命名为 <原文件名>__<工作表名>.csv。


[tests/README.md]
# Tests

可在此补充 Excel 转换的单测与回归样例。
