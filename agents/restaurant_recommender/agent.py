"""Restaurant recommender agent entry.

Recommends nearby restaurants by Dianping ratings and three price ranges,
excluding recently eaten food types.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import yaml


@dataclass(frozen=True)
class PriceRange:
    label: str
    min: float
    max: float

    def contains(self, price: float) -> bool:
        return self.min <= price <= self.max


@dataclass(frozen=True)
class RecentItem:
    name: str
    days_ago: float


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: str) -> str:
    return "".join(value.lower().split())


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _parse_price_ranges(raw: Any) -> list[PriceRange]:
    ranges = []
    for item in _to_list(raw):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        min_val = _safe_float(item.get("min"))
        max_val = _safe_float(item.get("max"))
        if min_val is None or max_val is None:
            continue
        ranges.append(PriceRange(label=label, min=min_val, max=max_val))
    return ranges


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _days_ago(value: Any, default_days: float) -> float:
    parsed = _parse_datetime(value)
    if parsed is None:
        return default_days
    now = datetime.now(timezone.utc)
    delta = now - parsed
    return max(delta.total_seconds() / 86400, 0.0)


def _normalize_recent_list(values: Any, default_days: float) -> list[RecentItem]:
    items = []
    for entry in _to_list(values):
        if isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("food") or entry.get("restaurant") or "").strip()
            if not name:
                continue
            days_ago = _days_ago(entry.get("ate_at"), default_days)
            items.append(RecentItem(name=name, days_ago=days_ago))
        else:
            name = str(entry).strip()
            if not name:
                continue
            items.append(RecentItem(name=name, days_ago=default_days))
    return items


def _fetch_supabase_rows(
    supabase_url: str,
    supabase_key: str,
    table: str,
    user_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    if not (supabase_url and supabase_key and table and user_id):
        return []
    query = urlencode(
        {
            "select": "*",
            "user_id": f"eq.{user_id}",
            "order": "ate_at.desc",
            "limit": str(limit),
        }
    )
    url = f"{supabase_url.rstrip('/')}/rest/v1/{table}?{query}"
    req = Request(
        url,
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        },
    )
    with urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, list) else []


def _resolve_env(value: str) -> str:
    if not value:
        return ""
    if value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    if value.startswith("$"):
        return os.getenv(value[1:], "")
    return value


def _load_recent_from_supabase(
    settings: dict[str, Any],
    user_id: str | None,
) -> tuple[list[RecentItem], list[RecentItem]]:
    if not user_id:
        return [], []
    supabase = settings.get("supabase") or {}
    supabase_cfg_path = Path(str(supabase.get("config_path") or "configs/supabase.yaml"))
    supabase_cfg = _load_yaml_config(supabase_cfg_path) if supabase_cfg_path.exists() else {}
    url = _resolve_env(str(supabase.get("url") or supabase_cfg.get("url") or ""))
    key = _resolve_env(
        str(
            supabase.get("service_role_key")
            or supabase_cfg.get("service_role_key")
            or supabase.get("key")
            or supabase_cfg.get("key")
            or supabase.get("anon_key")
            or supabase_cfg.get("anon_key")
            or ""
        )
    )
    tables = supabase.get("tables") or {}
    foods_table = str(tables.get("recent_foods") or "")
    restaurants_table = str(tables.get("recent_restaurants") or "")
    limit = int(supabase.get("recent_limit", 50))
    default_days = float(settings.get("recent_decay", {}).get("exclude_days", 0))

    try:
        foods_rows = _fetch_supabase_rows(url, key, foods_table, user_id, limit)
        rest_rows = _fetch_supabase_rows(url, key, restaurants_table, user_id, limit)
    except Exception:
        return [], []

    foods = _normalize_recent_list(foods_rows, default_days)
    restaurants = _normalize_recent_list(rest_rows, default_days)
    return foods, restaurants


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def _compute_distance_km(user: dict[str, Any], restaurant: dict[str, Any]) -> float | None:
    direct = _safe_float(restaurant.get("distance_km"))
    if direct is not None:
        return direct
    user_lat = _safe_float(user.get("lat"))
    user_lng = _safe_float(user.get("lng"))
    rest_lat = _safe_float(restaurant.get("lat"))
    rest_lng = _safe_float(restaurant.get("lng"))
    if None in (user_lat, user_lng, rest_lat, rest_lng):
        return None
    return _haversine_km(user_lat, user_lng, rest_lat, rest_lng)


def _find_recent_match(
    restaurant: dict[str, Any],
    recent_foods: list[RecentItem],
    recent_restaurants: list[RecentItem],
) -> tuple[str | None, float | None]:
    name = str(restaurant.get("name") or "")
    name_norm = _normalize_text(name)
    matched_days = None
    for item in recent_restaurants:
        if item.name and _normalize_text(item.name) in name_norm:
            matched_days = item.days_ago if matched_days is None else min(matched_days, item.days_ago)
    if matched_days is not None:
        return "recent_restaurant", matched_days

    fields = _to_str_list(restaurant.get("categories")) + _to_str_list(
        restaurant.get("dishes")
    )
    haystack = _normalize_text(" ".join(fields + [name]))
    for item in recent_foods:
        if item.name and _normalize_text(item.name) in haystack:
            matched_days = item.days_ago if matched_days is None else min(matched_days, item.days_ago)
    if matched_days is not None:
        return "recent_food", matched_days
    return None, None


def _score_restaurant(
    restaurant: dict[str, Any],
    distance_km: float | None,
    weights: dict[str, Any],
) -> float:
    rating = _safe_float(restaurant.get("rating")) or 0.0
    reviews = _safe_int(restaurant.get("review_count")) or 0
    rating_w = float(weights.get("rating", 1.0))
    reviews_w = float(weights.get("reviews", 0.3))
    distance_w = float(weights.get("distance", 0.4))
    score = rating_w * rating + reviews_w * math.log1p(reviews)
    if distance_km is not None:
        score += distance_w * (1 / (1 + distance_km))
    return score


def _price_fit_score(price: float, price_range: PriceRange) -> float:
    span = max(price_range.max - price_range.min, 1.0)
    mid = (price_range.max + price_range.min) / 2
    return max(0.0, 1 - abs(price - mid) / (span / 2))


def _party_fit_score(restaurant: dict[str, Any], party_size: int | None) -> float | None:
    if party_size is None:
        return None
    min_size = _safe_int(restaurant.get("party_size_min"))
    max_size = _safe_int(restaurant.get("party_size_max"))
    if min_size is None and max_size is None:
        return None
    if min_size is not None and party_size < min_size:
        return 0.0
    if max_size is not None and party_size > max_size:
        return 0.0
    return 1.0


def _decay_penalty(days_ago: float, half_life_days: float, weight: float) -> float:
    if days_ago <= 0:
        return weight
    if half_life_days <= 0:
        return weight
    decay = math.exp(-math.log(2) * days_ago / half_life_days)
    return weight * decay


def _extract_packages(restaurant: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def _select_menu(restaurant: dict[str, Any]) -> dict[str, Any] | None:
    menu_items = _to_str_list(restaurant.get("menu")) or _to_str_list(
        restaurant.get("dishes")
    )
    if menu_items:
        return {"type": "menu", "items": menu_items[:5]}
    dishes = _to_str_list(restaurant.get("dishes"))
    if dishes:
        return {"type": "dishes", "items": dishes[:5]}
    return None


def _amap_route_time(
    amap: dict[str, Any],
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> dict[str, Any] | None:
    api_key = str(amap.get("api_key") or "").strip()
    if not api_key:
        return None
    mode = str(amap.get("mode", "driving")).strip()
    strategy = str(amap.get("strategy", "10"))

    if mode == "walking":
        endpoint = "https://restapi.amap.com/v3/direction/walking"
    elif mode == "transit":
        endpoint = "https://restapi.amap.com/v3/direction/transit/integrated"
    else:
        endpoint = "https://restapi.amap.com/v3/direction/driving"

    params = {
        "key": api_key,
        "origin": f"{origin[1]},{origin[0]}",
        "destination": f"{destination[1]},{destination[0]}",
    }
    if mode == "driving":
        params["strategy"] = strategy
    query = urlencode(params)
    url = f"{endpoint}?{query}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    duration = None
    if mode == "transit":
        transits = payload.get("route", {}).get("transits", [])
        if transits:
            duration = _safe_int(transits[0].get("duration"))
    else:
        paths = payload.get("route", {}).get("paths", [])
        if paths:
            duration = _safe_int(paths[0].get("duration"))

    if duration is None:
        return None
    return {
        "provider": "amap",
        "mode": mode,
        "duration_seconds": duration,
        "duration_minutes": round(duration / 60),
    }


def _amap_place_around(
    amap: dict[str, Any],
    user: dict[str, Any],
    radius_km: float,
    keywords: str,
    types: str | None,
    page_size: int = 20,
    page: int = 1,
) -> list[dict[str, Any]]:
    api_key = str(amap.get("api_key") or "").strip()
    if not api_key:
        return []
    user_lat = _safe_float(user.get("lat"))
    user_lng = _safe_float(user.get("lng"))
    if user_lat is None or user_lng is None:
        return []
    radius_m = int(max(100, min(radius_km * 1000, 50000)))
    params = {
        "key": api_key,
        "location": f"{user_lng},{user_lat}",
        "keywords": keywords or "餐厅",
        "radius": str(radius_m),
        "sortrule": "distance",
        "offset": str(page_size),
        "page": str(page),
    }
    if types:
        params["types"] = types
    url = f"https://restapi.amap.com/v3/place/around?{urlencode(params)}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    pois = payload.get("pois") or []
    results: list[dict[str, Any]] = []
    for poi in pois:
        if not isinstance(poi, dict):
            continue
        location = str(poi.get("location") or "")
        if "," not in location:
            continue
        lng_str, lat_str = location.split(",", 1)
        lat = _safe_float(lat_str)
        lng = _safe_float(lng_str)
        biz_ext = poi.get("biz_ext") or {}
        rating = _safe_float(biz_ext.get("rating"))
        cost = _safe_float(biz_ext.get("cost"))
        categories = []
        if poi.get("type"):
            categories = [item for item in str(poi.get("type")).split(";") if item]
        results.append(
            {
                "id": poi.get("id"),
                "name": poi.get("name"),
                "lat": lat,
                "lng": lng,
                "price": cost,
                "rating": rating,
                "review_count": 0,
                "categories": categories,
                "address": poi.get("address"),
                "source": "amap",
                "distance_km": _safe_float(poi.get("distance")) / 1000 if _safe_float(poi.get("distance")) else None,
            }
        )
    return results


def _build_reasons(
    rating: float,
    reviews: int,
    distance_km: float | None,
) -> list[str]:
    reasons = []
    if rating >= 4.5:
        reasons.append("高评分")
    if reviews >= 300:
        reasons.append("评价量高")
    if distance_km is not None:
        reasons.append(f"距离约{distance_km:.1f}km")
    return reasons


def _prepare_settings(
    input_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = dict(config)
    for key in (
        "radius_km",
        "top_k_per_range",
        "min_rating",
        "min_reviews",
        "price_ranges",
        "weights",
    ):
        if key in input_data and input_data[key] is not None:
            settings[key] = input_data[key]
    return settings


def _load_amap_config(settings: dict[str, Any]) -> dict[str, Any]:
    amap_cfg = settings.get("amap") or {}
    amap_path = Path(str(amap_cfg.get("config_path") or "configs/amap.yaml"))
    file_cfg = _load_yaml_config(amap_path) if amap_path.exists() else {}
    merged = dict(file_cfg)
    for key, value in amap_cfg.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    api_key = _resolve_env(str(merged.get("api_key") or ""))
    merged["api_key"] = api_key
    return merged


def recommend_restaurants(
    input_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = _prepare_settings(input_data, config)
    price_ranges = _parse_price_ranges(settings.get("price_ranges"))
    if len(price_ranges) != 3:
        raise ValueError("price_ranges must include exactly 3 ranges.")

    radius_km = float(settings.get("radius_km", 3.0))
    top_k = int(settings.get("top_k_per_range", 5))
    min_rating = float(settings.get("min_rating", 0))
    min_reviews = int(settings.get("min_reviews", 0))
    weights = settings.get("weights") or {}
    price_mode = str(settings.get("price_mode", "per_person")).strip()
    decay_settings = settings.get("recent_decay") or {}
    exclude_days = float(decay_settings.get("exclude_days", 0))
    half_life_days = float(decay_settings.get("half_life_days", 7))

    user = input_data.get("user") or {}
    user_id = input_data.get("user_id") or user.get("id")
    recent_history = input_data.get("recent_history") or {}
    recent_foods = _normalize_recent_list(
        recent_history.get("foods") or input_data.get("recent_foods"),
        default_days=exclude_days,
    )
    recent_restaurants = _normalize_recent_list(
        recent_history.get("restaurants") or input_data.get("recent_restaurants"),
        default_days=exclude_days,
    )
    db_recent_foods, db_recent_restaurants = _load_recent_from_supabase(
        settings,
        str(user_id) if user_id else None,
    )
    recent_foods.extend(db_recent_foods)
    recent_restaurants.extend(db_recent_restaurants)

    restaurants = _to_list(input_data.get("restaurants"))
    if not restaurants:
        amap = _load_amap_config(settings)
        amap_keywords = str(input_data.get("amap_keywords") or amap.get("place_keywords") or "餐厅")
        amap_types = input_data.get("amap_types") or amap.get("place_types")
        restaurants = _amap_place_around(
            amap,
            user,
            radius_km=radius_km,
            keywords=amap_keywords,
            types=str(amap_types) if amap_types else None,
        )
    party_size = _safe_int(input_data.get("party_size") or user.get("party_size"))

    buckets: dict[str, list[dict[str, Any]]] = {r.label: [] for r in price_ranges}
    excluded_counts = {
        "recent_food": 0,
        "recent_restaurant": 0,
        "distance": 0,
        "rating": 0,
        "reviews": 0,
        "price": 0,
        "missing_location": 0,
    }

    for raw in restaurants:
        if not isinstance(raw, dict):
            continue
        distance_km = _compute_distance_km(user, raw)
        if distance_km is None:
            excluded_counts["missing_location"] += 1
            continue
        if distance_km > radius_km:
            excluded_counts["distance"] += 1
            continue

        rating = _safe_float(raw.get("rating")) or 0.0
        if rating < min_rating:
            excluded_counts["rating"] += 1
            continue

        reviews = _safe_int(raw.get("review_count")) or 0
        if reviews < min_reviews:
            excluded_counts["reviews"] += 1
            continue

        price = _safe_float(raw.get("price"))
        if price is None:
            excluded_counts["price"] += 1
            continue
        effective_price = price
        if price_mode == "total" and party_size:
            effective_price = price / max(party_size, 1)

        range_label = None
        selected_range = None
        for range_item in price_ranges:
            if range_item.contains(effective_price):
                range_label = range_item.label
                selected_range = range_item
                break
        if range_label is None:
            excluded_counts["price"] += 1
            continue

        matched_reason, matched_days = _find_recent_match(raw, recent_foods, recent_restaurants)
        if matched_reason and matched_days is not None and matched_days <= exclude_days:
            excluded_counts[matched_reason] += 1
            continue

        score = _score_restaurant(raw, distance_km, weights)
        if selected_range is not None:
            price_fit = _price_fit_score(effective_price, selected_range)
            score += float(weights.get("price_fit", 0.0)) * price_fit
        party_fit = _party_fit_score(raw, party_size)
        if party_fit is not None:
            score += float(weights.get("party_fit", 0.0)) * party_fit

        penalty = 0.0
        if matched_reason and matched_days is not None:
            penalty_weight = float(weights.get(matched_reason, 0.0))
            penalty = _decay_penalty(matched_days, half_life_days, penalty_weight)
            score -= penalty

        buckets[range_label].append(
            {
                "id": raw.get("id"),
                "name": raw.get("name"),
                "price": price,
                "effective_price": effective_price,
                "rating": rating,
                "review_count": reviews,
                "distance_km": distance_km,
                "categories": _to_str_list(raw.get("categories")),
                "dishes": _to_str_list(raw.get("dishes")),
                "address": raw.get("address"),
                "source": raw.get("source") or "dianping",
                "score": score,
                "penalty": penalty,
                "recent_match": matched_reason,
                "recent_days_ago": matched_days,
                "recommended_menu": _select_menu(raw),
                "lat": raw.get("lat"),
                "lng": raw.get("lng"),
                "reasons": _build_reasons(rating, reviews, distance_km),
            }
        )

    recommendations = []
    amap = _load_amap_config(settings)
    for range_item in price_ranges:
        items = sorted(
            buckets[range_item.label],
            key=lambda item: (item["score"], item["rating"], item["review_count"]),
            reverse=True,
        )
        top_items = items[:top_k]
        if top_items:
            origin_lat = _safe_float(user.get("lat"))
            origin_lng = _safe_float(user.get("lng"))
            if origin_lat is not None and origin_lng is not None:
                for item in top_items:
                    dest_lat = _safe_float(item.get("lat"))
                    dest_lng = _safe_float(item.get("lng"))
                    if dest_lat is None or dest_lng is None:
                        continue
                    route = _amap_route_time(
                        amap,
                        origin=(origin_lat, origin_lng),
                        destination=(dest_lat, dest_lng),
                    )
                    if route:
                        item["route"] = route
        recommendations.append(
            {
                "price_range": {
                    "label": range_item.label,
                    "min": range_item.min,
                    "max": range_item.max,
                },
                "restaurants": top_items,
            }
        )

    summary = {
        "radius_km": radius_km,
        "recent_foods_count": len(recent_foods),
        "recent_restaurants_count": len(recent_restaurants),
        "exclude_days": exclude_days,
        "party_size": party_size,
        "total_candidates": len(restaurants),
        "used_candidates": sum(len(items) for items in buckets.values()),
    }
    return {
        "input_summary": summary,
        "recommendations": recommendations,
        "excluded_counts": excluded_counts,
    }


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    input_path = payload.get("input_path")
    input_data = payload.get("input_data")
    if input_path:
        input_data = _load_json(Path(str(input_path)))
    if not isinstance(input_data, dict):
        raise ValueError("input_data must be provided as a JSON object.")

    config_path = Path(str(payload.get("config_path", "agents/restaurant_recommender/config/config.yaml")))
    config = _load_yaml_config(config_path)

    result = recommend_restaurants(input_data, config)
    output_path = payload.get("output_path")
    if output_path:
        path = Path(str(output_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_agent():
    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(_run)


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Recommend restaurants by location and price.")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument(
        "--config",
        default="agents/restaurant_recommender/config/config.yaml",
        help="Path to agent config YAML",
    )
    args = parser.parse_args()

    agent = build_agent()
    agent.invoke(
        {
            "input_path": args.input,
            "output_path": args.output,
            "config_path": args.config,
        }
    )


if __name__ == "__main__":
    _run_cli()
