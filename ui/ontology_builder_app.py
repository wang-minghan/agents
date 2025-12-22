import json
from pathlib import Path

import streamlit as st

from agents.ontology_builder.agent import build_all

st.set_page_config(page_title="Ontology Builder", layout="wide")

st.title("本体构建智能体")

with st.sidebar:
    st.header("输入配置")
    input_csv = st.text_input(
        "CSV 路径",
        value="input/本体对象关系列表.csv",
    )
    output_dir = st.text_input("输出目录", value="output")

st.write("使用 CSV 元数据构建本体对象与关系，输出设计文档与 OWL。")

if st.button("运行构建"):
    result = build_all(
        input_csv=Path(input_csv),
        output_dir=Path(output_dir),
    )
    st.success("构建完成")
    st.json(result)

    if Path(result["design_doc"]).exists():
        st.subheader("设计文档预览")
        st.markdown(Path(result["design_doc"]).read_text(encoding="utf-8"))
