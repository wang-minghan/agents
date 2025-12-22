from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.theme import inject_global_styles
from ui.views.excel_to_csv import render_excel_to_csv
from ui.views.food_picks import render_food_picks


BASE_DIR = Path.cwd()


def main() -> None:
    st.set_page_config(page_title="Food Picks", layout="wide")
    inject_global_styles()

    with st.sidebar:
        st.markdown("### Apps")
        page = st.radio("Navigate", ["Food Picks", "Excel to CSV"], label_visibility="collapsed")

    if page == "Food Picks":
        render_food_picks(BASE_DIR)
    else:
        render_excel_to_csv(BASE_DIR)


if __name__ == "__main__":
    main()
