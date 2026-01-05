from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from google import genai


def _resolve_api_key() -> Optional[str]:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def generate_image(
    prompt: str,
    output_path: Path,
    model: str = "gemini-2.5-flash-image",
    api_key: Optional[str] = None,
) -> dict:
    api_key = api_key or _resolve_api_key()
    if not api_key:
        return {"status": "failed", "reason": "missing_api_key"}

    if not model:
        model = os.environ.get("NANOBANNA_MODEL", "gemini-2.5-flash-image")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    image = None
    for part in getattr(response, "parts", []) or []:
        if getattr(part, "inline_data", None):
            image = part.as_image()
            break

    if image is None:
        return {"status": "failed", "reason": "no_image_returned"}

    image.save(output_path)
    return {"status": "completed", "path": str(output_path)}
