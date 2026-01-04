# Streamlit 使用说明

## 安装依赖

```bash
poetry install
```

## 启动

```bash
poetry run streamlit run ui/streamlit_app.py
```

默认地址：`http://localhost:8501`

## 入口说明

- 默认首页为聚合入口（Agents Hub），动态展示可用 Agent 页面
- 侧边栏可直接进入具体 Agent 页面
- Dev Team 页面提供“发布到 Excel Agent”按钮，确认后全量覆盖 `agents/excel_to_csv`
