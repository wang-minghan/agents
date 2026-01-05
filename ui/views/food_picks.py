from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.request import Request, urlopen

import streamlit as st
import streamlit.components.v1 as components
import yaml

from agents.restaurant_recommender.agent import recommend_restaurants
from ui.theme.theme import hero_html, poster_palette, price_section_html

COMPONENT_DIR = Path(__file__).resolve().parent.parent / "components" / "geolocation"
geolocation_component = components.declare_component("geolocation", path=str(COMPONENT_DIR))


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _supabase_insert(config_path: Path, table: str, rows: list[dict]) -> tuple[bool, str | None]:
    cfg = _load_yaml(config_path)
    url = str(cfg.get("url") or "")
    key = str(cfg.get("service_role_key") or cfg.get("key") or cfg.get("anon_key") or "")
    if not (url and key and table and rows):
        return False, "missing config or payload"
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
    req = Request(
        endpoint,
        method="POST",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        data=json.dumps(rows).encode("utf-8"),
    )
    try:
        with urlopen(req, timeout=10) as response:
            response.read()
    except HTTPError as err:
        detail = None
        try:
            payload = json.loads(err.read().decode("utf-8", "ignore") or "{}")
            detail = payload.get("message") or payload.get("error")
        except Exception:
            detail = None
        if detail:
            return False, f"http {err.code}: {detail}"
        return False, f"http {err.code}"
    except URLError as err:
        return False, f"url error: {err.reason}"
    except Exception:
        return False, "unknown error"
    return True, None


def _load_amap_key(config_path: Path) -> str:
    cfg = _load_yaml(config_path)
    return str(cfg.get("api_key") or "").strip()


def _amap_ip_location(config_path: Path) -> tuple[float | None, float | None, str | None]:
    api_key = _load_amap_key(config_path)
    if not api_key:
        return None, None, None
    url = f"https://restapi.amap.com/v3/ip?key={api_key}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None, None, None
    rectangle = str(payload.get("rectangle") or "")
    if ";" not in rectangle:
        return None, None, payload.get("city")
    first, second = rectangle.split(";", 1)
    if "," not in first or "," not in second:
        return None, None, payload.get("city")
    lng1, lat1 = first.split(",", 1)
    lng2, lat2 = second.split(",", 1)
    try:
        lat = (float(lat1) + float(lat2)) / 2
        lng = (float(lng1) + float(lng2)) / 2
    except ValueError:
        return None, None, payload.get("city")
    return lat, lng, payload.get("city")


def _get_param(params: dict, key: str) -> str | None:
    value = params.get(key)
    if isinstance(value, list):
        return value[0]
    return value

def _get_browser_location() -> dict | None:
    return geolocation_component(enableHighAccuracy=True, timeout=15000, maximumAge=0, key="geo")


def _poster_card_html(item: dict) -> str:
    name = escape(str(item.get("name") or "Unknown"))
    rating = item.get("rating")
    price = item.get("price")
    distance = item.get("distance_km")
    route = item.get("route") or {}
    duration = route.get("duration_minutes")
    address = escape(str(item.get("address") or ""))
    menu = item.get("recommended_menu") or {}
    menu_items = menu.get("items") or []

    palette = poster_palette(name)
    rating_html = f'<div class="poster-badge">★ {rating}</div>' if rating is not None else ""
    chip_text = f"¥{price}" if price is not None else "Curated"
    chip_html = f'<div class="poster-chip">{escape(chip_text)}</div>' if chip_text else ""

    tags = []
    if price is not None:
        tags.append(f"¥{price}")
    if distance is not None:
        tags.append(f"{distance:.1f} km")
    if duration is not None:
        tags.append(f"{duration} min")
    if rating is not None:
        tags.append(f"Rating {rating}")
    tags_html = "".join(f"<span>{escape(tag)}</span>" for tag in tags)

    address_html = f'<div class="poster-meta">{address}</div>' if address else ""
    menu_html = (
        f'<div class="poster-meta">Menu: {escape("、".join(menu_items))}</div>'
        if menu_items
        else ""
    )

    return (
        f'<div class="poster-card" style="--poster-a: {palette[0]}; '
        f'--poster-b: {palette[1]}; --poster-c: {palette[2]};">'
        "<div class=\"poster-art\">"
        '<div class="poster-overlay"></div>'
        f"{rating_html}{chip_html}<div class=\"poster-title\">{name}</div>"
        "</div>"
        "<div class=\"poster-body\">"
        f"<div class=\"poster-tags\">{tags_html}</div>{address_html}{menu_html}"
        "</div></div>"
    )


def _location_line(
    lat: float,
    lng: float,
    source: str,
    radius_km: float,
    party_size: int,
    accuracy: float | None = None,
) -> str:
    source_label = "Browser" if source == "browser" else "IP" if source == "ip" else "Manual"
    accuracy_label = f" · ±{accuracy:.0f}m" if accuracy else ""
    return (
        f'<div class="poster-meta">Location: {source_label} '
        f"· {lat:.5f}, {lng:.5f}{accuracy_label} · Radius {radius_km} km · Party {party_size}</div>"
    )


def render_food_picks(base_dir: Path) -> None:
    supabase_config = base_dir / "agents" / "restaurant_recommender" / "config" / "supabase.yaml"
    amap_config = base_dir / "agents" / "restaurant_recommender" / "config" / "amap.yaml"
    agent_config = base_dir / "agents" / "restaurant_recommender" / "config" / "config.yaml"

    saved = st.session_state.get("food_saved", False)
    stage = "save" if saved else "locate"
    hero_slot = st.empty()
    hero_slot.markdown(hero_html(stage), unsafe_allow_html=True)
    status_slot = st.empty()

    def set_status(text: str) -> None:
        status_slot.markdown(
            f'<div class="status-shell"><div class="status-pill">{escape(text)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Nearby picks</div>', unsafe_allow_html=True)
    set_status("Locating…")

    params = st.query_params
    lat_param = _get_param(params, "lat")
    lng_param = _get_param(params, "lng")
    user_id = _get_param(params, "user_id") or "u-001"
    radius_km = float(_get_param(params, "radius_km") or 2.0)
    party_size = int(_get_param(params, "party_size") or 2)
    keywords = _get_param(params, "amap_keywords")
    types = _get_param(params, "amap_types")

    lat = float(lat_param) if lat_param else None
    lng = float(lng_param) if lng_param else None
    source = None
    accuracy = None

    if lat is not None and lng is not None:
        source = "manual"

    geo_error = None
    if lat is None or lng is None:
        cached_precise = st.session_state.get("precise_location")
        if cached_precise:
            lat = cached_precise.get("lat")
            lng = cached_precise.get("lng")
            accuracy = cached_precise.get("accuracy")
            source = "browser"
        else:
            set_status("Requesting precise location…")
            geo = _get_browser_location()
            if geo and isinstance(geo, dict):
                status = str(geo.get("status") or "")
                if status == "success":
                    lat = geo.get("lat")
                    lng = geo.get("lng")
                    accuracy = geo.get("accuracy")
                    st.session_state["precise_location"] = {
                        "lat": lat,
                        "lng": lng,
                        "accuracy": accuracy,
                    }
                    source = "browser"
                elif status:
                    geo_error = status

    if (lat is None or lng is None) and geo_error in {"denied", "timeout", "unavailable"}:
        cached = st.session_state.get("ip_location")
        if cached:
            lat, lng, _ = cached
        else:
            ip_lat, ip_lng, city = _amap_ip_location(amap_config)
            st.session_state["ip_location"] = (ip_lat, ip_lng, city)
            lat, lng = ip_lat, ip_lng
        if lat is not None and lng is not None:
            source = "ip"
            set_status("Using IP location fallback.")

    if lat is None or lng is None:
        st.markdown('<div class="status-note">等待定位完成…</div>', unsafe_allow_html=True)
        st.stop()

    if not saved:
        stage = "recommend"
    hero_slot.markdown(hero_html(stage), unsafe_allow_html=True)
    set_status("Generating recommendations…")

    payload = {
        "user_id": user_id,
        "party_size": party_size,
        "radius_km": radius_km,
        "amap_keywords": keywords,
        "amap_types": types,
        "user": {"lat": lat, "lng": lng},
    }
    config = _load_yaml(agent_config)
    result = recommend_restaurants(payload, config)
    st.session_state["rr_result"] = result

    if not saved:
        stage = "choose"
    hero_slot.markdown(hero_html(stage), unsafe_allow_html=True)
    set_status("Ready to pick.")
    st.markdown(
        _location_line(lat, lng, source or "browser", radius_km, party_size, accuracy),
        unsafe_allow_html=True,
    )

    if not result:
        st.warning("No recommendations returned.")
        return

    recommendations = result.get("recommendations", [])
    if not recommendations:
        st.info("No restaurants found for the current filters.")
        return

    for block in recommendations:
        pr = block.get("price_range") or {}
        st.markdown(
            price_section_html(pr.get("label") or "Budget", pr.get("min"), pr.get("max")),
            unsafe_allow_html=True,
        )
        restaurants = block.get("restaurants", [])
        if not restaurants:
            st.caption("No matches in this price range.")
            continue
        cols = st.columns(2, gap="large")
        for idx, item in enumerate(restaurants):
            with cols[idx % 2]:
                st.markdown(_poster_card_html(item), unsafe_allow_html=True)
                action_cols = st.columns([3, 1])
                with action_cols[1]:
                    if st.button("Pick", key=f"select-{pr.get('label')}-{idx}"):
                        set_status("Saving selection…")
                        now = datetime.now(timezone.utc).isoformat()
                        restaurant_row = {
                            "user_id": user_id,
                            "restaurant": item.get("name"),
                            "ate_at": now,
                        }
                        ok_restaurant, err_restaurant = _supabase_insert(
                            supabase_config,
                            "recent_restaurants",
                            [restaurant_row],
                        )
                        food_rows = [
                            {"user_id": user_id, "food": food, "ate_at": now}
                            for food in (item.get("recommended_menu") or {}).get("items") or []
                        ]
                        ok_foods = True
                        err_foods = None
                        if food_rows:
                            ok_foods, err_foods = _supabase_insert(
                                supabase_config,
                                "recent_foods",
                                food_rows,
                            )
                        st.session_state["food_saved"] = True
                        hero_slot.markdown(hero_html("save"), unsafe_allow_html=True)
                        set_status("Saved.")
                        if ok_restaurant and ok_foods:
                            st.success("Saved to history")
                        else:
                            st.warning("Saved locally, but Supabase insert failed.")
                            if err_restaurant:
                                st.caption(f"recent_restaurants: {err_restaurant}")
                            if err_foods:
                                st.caption(f"recent_foods: {err_foods}")
