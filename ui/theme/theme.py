from __future__ import annotations

import hashlib
from html import escape

import streamlit as st


STAGES = (
    ("locate", "Locate"),
    ("recommend", "Recommend"),
    ("choose", "Choose"),
    ("save", "Save"),
)

PALETTES = (
    ("#ff8a65", "#ffb74d", "#ffe0b2"),
    ("#7e57c2", "#64b5f6", "#b3e5fc"),
    ("#26a69a", "#80cbc4", "#c8e6c9"),
    ("#ef5350", "#f06292", "#f8bbd0"),
    ("#5c6bc0", "#7986cb", "#c5cae9"),
    ("#8d6e63", "#b39ddb", "#d7ccc8"),
)


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Archivo:wght@400;500;600&display=swap');

          :root {
            --ink: #1b1b1f;
            --muted: #4b4a52;
            --accent: #ff6f3d;
            --accent-2: #2f8f83;
            --paper: #f8f2e8;
            --panel: rgba(255, 255, 255, 0.78);
            --shadow: 0 18px 45px rgba(16, 15, 20, 0.12);
          }

          .stApp {
            background:
              radial-gradient(circle at 12% 18%, rgba(255, 185, 120, 0.35), transparent 45%),
              radial-gradient(circle at 88% 12%, rgba(111, 196, 186, 0.35), transparent 42%),
              linear-gradient(180deg, #f9f4ec 0%, #f1eadf 100%);
            color: var(--ink);
            font-family: "Archivo", sans-serif;
          }

          header, footer, #MainMenu {
            visibility: hidden;
          }

          .block-container {
            padding-top: 2.8rem;
            padding-bottom: 4rem;
          }

          div[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.85);
            border-right: 1px solid rgba(0, 0, 0, 0.06);
          }

          .hero-card {
            display: grid;
            gap: 0.9rem;
            align-items: start;
            padding: 2.2rem 2.8rem;
            background: var(--panel);
            border-radius: 30px;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(14px);
            animation: fadeUp 0.8s ease forwards;
          }

          .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.25rem;
            font-size: 0.68rem;
            color: var(--muted);
          }

          .hero-title {
            font-family: "Cormorant Garamond", serif;
            font-size: clamp(2.1rem, 3.2vw, 3.5rem);
            line-height: 1.05;
            margin: 0;
          }

          .hero-sub {
            color: var(--muted);
            font-size: 1rem;
            max-width: 42rem;
          }

          .stage-bar {
            margin-top: 0.8rem;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.2rem;
          }

          .stage-pill {
            border-radius: 999px;
            padding: 0.38rem 0.7rem;
            text-align: center;
            font-size: 0.68rem;
            letter-spacing: 0.12rem;
            text-transform: uppercase;
            background: transparent;
            border-bottom: 2px solid transparent;
            color: var(--muted);
          }

          .stage-pill.active {
            color: var(--ink);
            border-bottom-color: var(--accent);
            box-shadow: none;
          }

          .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(0, 0, 0, 0.08);
            color: var(--muted);
            font-size: 0.78rem;
            box-shadow: 0 8px 18px rgba(16, 15, 20, 0.08);
          }

          .status-shell {
            min-height: 2.4rem;
            margin-top: 0.6rem;
            display: flex;
            align-items: center;
          }

          .status-note {
            margin-top: 0.8rem;
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(0, 0, 0, 0.08);
            color: var(--muted);
          }

          .section-title {
            font-family: "Cormorant Garamond", serif;
            font-size: 1.6rem;
            margin: 2rem 0 0.6rem;
          }

          .price-head {
            display: flex;
            align-items: baseline;
            gap: 0.8rem;
            margin: 1.8rem 0 1rem;
          }

          .price-head h3 {
            font-family: "Cormorant Garamond", serif;
            font-size: 1.45rem;
            margin: 0;
          }

          .price-range {
            font-size: 0.75rem;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            background: rgba(47, 143, 131, 0.12);
            color: var(--accent-2);
            letter-spacing: 0.08rem;
            text-transform: uppercase;
          }

          .poster-card {
            background: rgba(255, 255, 255, 0.92);
            border-radius: 24px;
            box-shadow: var(--shadow);
            overflow: hidden;
            border: 1px solid rgba(0, 0, 0, 0.06);
            margin-bottom: 1.5rem;
            animation: fadeUp 0.7s ease forwards;
          }

          .poster-art {
            height: 240px;
            position: relative;
            background:
              radial-gradient(circle at 20% 20%, var(--poster-c), transparent 45%),
              linear-gradient(135deg, var(--poster-a), var(--poster-b));
            overflow: hidden;
          }

          .poster-overlay {
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(0, 0, 0, 0.05), rgba(0, 0, 0, 0.55));
          }

          .poster-title {
            position: absolute;
            left: 1.2rem;
            bottom: 1rem;
            color: #fff;
            font-family: "Cormorant Garamond", serif;
            font-size: 1.4rem;
            z-index: 2;
          }

          .poster-badge {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: rgba(255, 255, 255, 0.85);
            color: var(--ink);
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            z-index: 2;
          }

          .poster-chip {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(27, 27, 31, 0.65);
            color: #fff;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            font-size: 0.7rem;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
            z-index: 2;
          }

          .poster-body {
            padding: 1.1rem 1.4rem 1.2rem;
            display: grid;
            gap: 0.45rem;
          }

          .poster-meta {
            color: var(--muted);
            font-size: 0.85rem;
          }

          .poster-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
          }

          .poster-tags span {
            font-size: 0.72rem;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: rgba(0, 0, 0, 0.06);
            color: var(--muted);
          }

          .stButton > button {
            background: var(--accent);
            color: #fff;
            border-radius: 999px;
            border: none;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            box-shadow: 0 10px 20px rgba(255, 111, 61, 0.35);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
          }

          .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(255, 111, 61, 0.45);
          }

          .stTextInput input, .stNumberInput input {
            border-radius: 14px;
            border: 1px solid rgba(0, 0, 0, 0.08);
            padding: 0.55rem 0.75rem;
            background: rgba(255, 255, 255, 0.9);
          }

          .stProgress > div > div {
            background-image: linear-gradient(90deg, var(--accent), var(--accent-2));
          }

          @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_html(stage: str) -> str:
    pills = []
    for key, label in STAGES:
        active = " active" if key == stage else ""
        pills.append(f'<div class="stage-pill{active}">{escape(label)}</div>')
    return """
    <div class="hero-card">
      <div class="hero-kicker">Tonight, curated</div>
      <h1 class="hero-title">Food Picks</h1>
      <div class="hero-sub">Auto-locate, curate nearby hits, and keep your dining history clean. Tap once to save.</div>
      <div class="stage-bar">{}</div>
    </div>
    """.format("".join(pills))


def price_section_html(label: str, min_price: float | int | None, max_price: float | int | None) -> str:
    safe_label = escape(label or "Budget")
    min_value = "?" if min_price is None else min_price
    max_value = "?" if max_price is None else max_price
    return (
        f"<div class=\"price-head\"><h3>{safe_label}</h3>"
        f"<span class=\"price-range\">¥{min_value} - ¥{max_value}</span></div>"
    )


def poster_palette(seed: str) -> tuple[str, str, str]:
    if not seed:
        return PALETTES[0]
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(PALETTES)
    return PALETTES[idx]
