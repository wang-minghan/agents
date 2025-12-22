from __future__ import annotations

RELATION_DEFS = {
    "includes": {
        "label": "包含",
        "inverse_id": "isPartOf",
        "inverse_label": "属于",
    },
    "relatesTo": {
        "label": "关联",
        "inverse_id": "isRelatedTo",
        "inverse_label": "被关联",
    },
    "affects": {
        "label": "影响",
        "inverse_id": "isAffectedBy",
        "inverse_label": "被影响",
    },
    "constitutes": {
        "label": "构成",
        "inverse_id": "splitsInto",
        "inverse_label": "拆分",
    },
}

RELATION_LABEL_TO_ID = {
    "包含": "includes",
    "关联": "relatesTo",
    "影响": "affects",
    "构成": "constitutes",
}

RELATION_CODE_TO_ID = {
    "includes": "includes",
    "relatesto": "relatesTo",
    "affects": "affects",
    "constitutes": "constitutes",
}
