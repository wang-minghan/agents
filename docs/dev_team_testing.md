结论：dev_team 自动执行单元测试，默认单次运行并启用失败早停（-x），无需配置。

行为
- 自动发现 `test_*.py` 与 `*_test.py`
- 默认使用 `pytest -x` 针对目标目录运行
- 未发现测试时返回 SKIPPED
