from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.theme.theme import inject_global_styles
from ui.views.dev_team import render_dev_team
from ui.views.excel_to_csv import render_excel_to_csv
from ui.views.food_picks import render_food_picks
from ui.views.task_planner import render_task_planner


BASE_DIR = PROJECT_ROOT


def main() -> None:
    st.set_page_config(page_title="Food Picks", layout="wide")
    inject_global_styles()

    pages = [
        ("Food Picks", "food"),
        ("Excel to CSV", "excel"),
        ("Task Planner", "planner"),
        ("Dev Team", "dev"),
    ]
    label_by_slug = {slug: label for label, slug in pages}
    slug_by_label = {label: slug for label, slug in pages}
    query_page = st.query_params.get("page", "food")
    if isinstance(query_page, list):
        query_page = query_page[0]
    default_label = label_by_slug.get(query_page, "Food Picks")
    default_index = [label for label, _ in pages].index(default_label)

    with st.sidebar:
        st.markdown("### Apps")
        page = st.radio(
            "Navigate",
            [label for label, _ in pages],
            label_visibility="collapsed",
            index=default_index,
        )

    selected_slug = slug_by_label.get(page, "food")
    if query_page != selected_slug:
        st.query_params["page"] = selected_slug

    if page == "Food Picks":
        render_food_picks(BASE_DIR)
    elif page == "Excel to CSV":
        render_excel_to_csv(BASE_DIR)
    elif page == "Task Planner":
        render_task_planner(BASE_DIR)
    else:
        render_dev_team(BASE_DIR)


if __name__ == "__main__":
    main()
