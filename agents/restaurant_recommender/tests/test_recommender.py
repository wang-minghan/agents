from agents.restaurant_recommender.agent import recommend_restaurants


def _config():
    return {
        "radius_km": 3.0,
        "top_k_per_range": 5,
        "min_rating": 0.0,
        "min_reviews": 0,
        "price_mode": "per_person",
        "price_ranges": [
            {"label": "budget", "min": 0, "max": 50},
            {"label": "mid", "min": 50, "max": 120},
            {"label": "premium", "min": 120, "max": 9999},
        ],
        "weights": {
            "rating": 1.0,
            "reviews": 0.0,
            "distance": 0.0,
            "price_fit": 0.0,
            "party_fit": 0.0,
            "recent_food": 1.0,
            "recent_restaurant": 1.0,
        },
        "recent_decay": {"exclude_days": 1, "half_life_days": 7},
    }


def _base_input():
    return {
        "user": {"lat": 0, "lng": 0},
        "recent_foods": [],
        "recent_restaurants": [],
        "restaurants": [],
    }


def test_excludes_recent_foods():
    data = _base_input()
    data["recent_foods"] = ["火锅"]
    data["restaurants"] = [
        {
            "name": "老张火锅",
            "distance_km": 1.0,
            "price": 80,
            "rating": 4.5,
            "review_count": 100,
            "categories": ["火锅"],
        },
        {
            "name": "阿美餐厅",
            "distance_km": 1.0,
            "price": 80,
            "rating": 4.6,
            "review_count": 120,
            "categories": ["本帮菜"],
        },
    ]

    result = recommend_restaurants(data, _config())
    mid_range = next(
        item for item in result["recommendations"] if item["price_range"]["label"] == "mid"
    )
    names = [item["name"] for item in mid_range["restaurants"]]
    assert "阿美餐厅" in names
    assert "老张火锅" not in names


def test_groups_by_price_range():
    data = _base_input()
    data["restaurants"] = [
        {
            "name": "平价小馆",
            "distance_km": 1.0,
            "price": 30,
            "rating": 4.0,
            "review_count": 50,
        },
        {
            "name": "中档餐厅",
            "distance_km": 1.0,
            "price": 80,
            "rating": 4.2,
            "review_count": 60,
        },
        {
            "name": "高端餐厅",
            "distance_km": 1.0,
            "price": 200,
            "rating": 4.8,
            "review_count": 200,
        },
    ]

    result = recommend_restaurants(data, _config())
    counts = {
        item["price_range"]["label"]: len(item["restaurants"])
        for item in result["recommendations"]
    }
    assert counts["budget"] == 1
    assert counts["mid"] == 1
    assert counts["premium"] == 1


def test_filters_by_radius():
    data = _base_input()
    data["restaurants"] = [
        {
            "name": "远处餐厅",
            "distance_km": 10.0,
            "price": 80,
            "rating": 4.2,
            "review_count": 60,
        }
    ]

    result = recommend_restaurants(data, _config())
    used = result["input_summary"]["used_candidates"]
    assert used == 0
    assert result["excluded_counts"]["distance"] == 1
