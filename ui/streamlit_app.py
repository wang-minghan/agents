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
from ui.views.architect import render_architect


BASE_DIR = PROJECT_ROOT


def _render_index(apps: list[dict[str, str]]) -> None:
    st.markdown('<div class="section-title">Agents Hub</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <style>
          .hub-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.2rem;
            margin-top: 1rem;
          }
          .hub-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: 22px;
            padding: 1.2rem 1.1rem;
            box-shadow: 0 16px 32px rgba(16, 15, 20, 0.12);
          }
          .hub-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18rem;
            font-size: 0.62rem;
            color: #6d6a77;
          }
          .hub-title {
            margin-top: 0.45rem;
            font-size: 1.1rem;
            font-weight: 700;
            color: #1b1b1f;
          }
          .hub-meta {
            margin-top: 0.5rem;
            font-size: 0.85rem;
            color: #595562;
          }
          .hub-link {
            display: inline-block;
            margin-top: 0.8rem;
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            color: #1b1b1f;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
          }
          .hub-link:hover {
            border-color: rgba(0, 0, 0, 0.25);
            box-shadow: 0 8px 16px rgba(16, 15, 20, 0.12);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cards = []
    for app in apps:
        cards.append(
            f"""
            <div class="hub-card">
              <div class="hub-kicker">{app['status']}</div>
              <div class="hub-title">{app['label']}</div>
              <div class="hub-meta">{app['path']}</div>
              <a class="hub-link" href="?page={app['slug']}">进入</a>
            </div>
            """
        )
    st.markdown(f"<div class='hub-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Agents Hub", layout="wide")
    inject_global_styles()

    apps = [
        {
            "label": "Dev Team",
            "slug": "dev",
            "render": render_dev_team,
            "path": "agents/dev_team",
        },
        {
            "label": "架构师",
            "slug": "architect",
            "render": render_architect,
            "path": "agents/dev_team/architect",
        },
        {
            "label": "Excel to CSV",
            "slug": "excel",
            "render": render_excel_to_csv,
            "path": "agents/excel_to_csv",
        },
        {
            "label": "Food Picks",
            "slug": "food",
            "render": render_food_picks,
            "path": "agents/restaurant_recommender",
        },
    ]
    label_by_slug = {app["slug"]: app["label"] for app in apps}
    slug_by_label = {app["label"]: app["slug"] for app in apps}
    query_page = st.query_params.get("page", "")
    if isinstance(query_page, list):
        query_page = query_page[0]
    default_label = label_by_slug.get(query_page, "首页")
    page_labels = ["首页"] + [app["label"] for app in apps]
    default_index = page_labels.index(default_label)

    with st.sidebar:
        st.markdown("### Apps")
        page = st.radio(
            "Navigate",
            page_labels,
            label_visibility="collapsed",
            index=default_index,
        )

    if page == "首页":
        cards = []
        for app in apps:
            cards.append(
                {
                    "label": app["label"],
                    "slug": app["slug"],
                    "path": app["path"],
                    "status": "READY" if (BASE_DIR / app["path"]).exists() else "MISSING",
                }
            )
        _render_index(cards)
        if query_page:
            st.query_params["page"] = ""
        return

    selected_slug = slug_by_label.get(page, "")
    if query_page != selected_slug:
        st.query_params["page"] = selected_slug

    for app in apps:
        if app["label"] == page:
            app["render"](BASE_DIR)
            break


if __name__ == "__main__":
    main()
