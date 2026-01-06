from __future__ import annotations

from typing import Protocol, Any


class Policy(Protocol):
    def run(self, **kwargs: Any) -> Any: ...
