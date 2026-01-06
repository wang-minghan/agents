from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from agents.dev_team.memory import SharedMemoryStore


@dataclass
class CompliancePolicyImpl:
    config: Dict[str, Any]
    output_dir: Path
    memory: SharedMemoryStore

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        cfg = self.config.get("compliance", {})
        if cfg.get("enabled", True) is False:
            return {"status": "skipped", "reason": "disabled"}

        scan_root = self._resolve_scan_root(cfg)
        if not scan_root.exists():
            return {"status": "skipped", "reason": "scan_root_missing", "scan_root": str(scan_root)}
        if not scan_root.is_dir():
            return {"status": "skipped", "reason": "scan_root_not_dir", "scan_root": str(scan_root)}
        patterns = cfg.get("secret_patterns") or [
            r"sk-[A-Za-z0-9]{20,}",
            r"AIza[0-9A-Za-z\-_]{10,}",
            r"(?i)api[_-]?key\\s*[:=]\\s*['\\\"]?[A-Za-z0-9\\-_=]{16,}",
            r"(?i)secret\\s*[:=]\\s*['\\\"]?[A-Za-z0-9\\-_=]{12,}",
            r"(?i)token\\s*[:=]\\s*['\\\"]?[A-Za-z0-9\\-_=]{12,}",
        ]
        deny_globs = cfg.get("deny_globs", [])
        skip_dirs = set(cfg.get("skip_dirs", []))
        max_bytes = int(cfg.get("max_bytes", 200_000))

        findings = self._scan_for_secrets(
            scan_root,
            patterns=patterns,
            deny_globs=deny_globs,
            skip_dirs=skip_dirs,
            max_bytes=max_bytes,
        )
        warnings: List[str] = []
        if self.config.get("execution", {}).get("allow_unsafe", True) and cfg.get(
            "require_safe_execution", False
        ):
            warnings.append("unsafe_execution_enabled")

        status = "failed" if findings else "passed"
        summary = f"findings={len(findings)}"
        report = {
            "status": status,
            "summary": summary,
            "scan_root": str(scan_root),
            "findings": findings,
            "warnings": warnings,
        }
        self.memory.global_context["compliance_report"] = report
        return report

    def _resolve_scan_root(self, cfg: Dict[str, Any]) -> Path:
        scan_root = cfg.get("scan_root")
        if scan_root:
            candidate = Path(scan_root).expanduser()
            if not candidate.is_absolute():
                candidate = (self.output_dir / candidate).resolve()
            return candidate
        return self.output_dir

    def _scan_for_secrets(
        self,
        root: Path,
        *,
        patterns: List[str],
        deny_globs: List[str],
        skip_dirs: set[str],
        max_bytes: int,
    ) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        regexes = [re.compile(pattern) for pattern in patterns]
        default_skip = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in default_skip and d not in skip_dirs]
            for filename in filenames:
                path = Path(dirpath) / filename
                if deny_globs and any(path.match(glob) for glob in deny_globs):
                    continue
                try:
                    if path.stat().st_size > max_bytes:
                        continue
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for regex in regexes:
                    match = regex.search(content)
                    if match:
                        snippet = match.group(0)
                        findings.append(
                            {
                                "path": str(path),
                                "pattern": regex.pattern,
                                "sample": snippet[:64],
                            }
                        )
                        break
        return findings
