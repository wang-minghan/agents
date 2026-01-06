from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import shutil
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.common import find_project_root
from agents.dev_team.memory import SharedMemoryStore


@dataclass
class UIBaselinePolicy:
    config: Dict[str, Any]
    output_dir: Path
    memory: SharedMemoryStore
    _frontend_detected: Optional[bool] = field(default=None, init=False)

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        requirements = kwargs.get("requirements")
        return self.prepare_assets(requirements)

    def allow_missing_baseline(self) -> bool:
        return self.config.get("ui_design", {}).get("allow_without_baseline", True)

    def requires_ui_baseline_from_requirements(self, requirements: Any) -> bool:
        text = str(requirements).lower()
        keywords = ("前端", "ui", "界面", "页面", "网页", "frontend", "web", "design")
        if any(keyword in text for keyword in keywords):
            return True
        return self._should_force_ui_design()

    def requires_ui_baseline(self) -> bool:
        requirements = self.memory.global_context.get("requirements", "")
        return self.requires_ui_baseline_from_requirements(requirements)

    def requires_ui_functional_audit(self, requirements: Any) -> bool:
        ui_cfg = self.config.get("ui_design", {})
        if ui_cfg.get("enabled", True) is False:
            return False
        if isinstance(requirements, dict) and requirements.get("ui_required"):
            return True
        if self.requires_ui_baseline_from_requirements(requirements):
            return True
        return self._detect_frontend_presence(self.output_dir)

    def baseline_required(self) -> bool:
        return self.memory.global_context.get("ui_design_source") == "user"

    def user_baseline_configured(self) -> bool:
        ui_cfg = self.config.get("ui_design", {})
        if ui_cfg.get("baseline_path") or os.environ.get("UI_DESIGN_BASELINE"):
            return True
        return False

    def should_force_ui_design(self) -> bool:
        return self._should_force_ui_design()

    def detect_frontend_presence(self, root: Path, max_files: int = 300) -> bool:
        return self._detect_frontend_presence(root, max_files=max_files)

    def ui_comparison_required(self) -> bool:
        return self._ui_comparison_required()

    def ui_diff_threshold(self) -> float:
        return self._ui_diff_threshold()

    def ui_layout_threshold(self) -> float:
        return self._ui_layout_threshold()

    def ui_edge_threshold(self) -> int:
        return self._ui_edge_threshold()

    def ui_layout_compare_size(self) -> tuple[int, int]:
        return self._ui_layout_compare_size()

    def build_ui_design_prompt(self, requirements: Dict[str, Any]) -> str:
        return self._build_ui_design_prompt(requirements)

    def build_nanobanna_command(self, prompt: str, output_path: Path) -> List[str]:
        return self._build_nanobanna_command(prompt, output_path)

    def run_nanobanna(self, prompt: str, output_path: Path) -> Dict[str, Any]:
        return self._run_nanobanna(prompt, output_path)

    def resolve_user_baseline(self, evidence_dir: Path) -> Optional[Path]:
        return self._resolve_user_baseline(evidence_dir)

    def summarize_ui_baseline(self, baseline: Path, requirements: Dict[str, Any]) -> Dict[str, Any]:
        return self._summarize_ui_baseline(baseline, requirements)

    def check_ui_evidence(self) -> Dict[str, Any]:
        evidence_dir = self.output_dir / "evidence" / "ui"
        baseline = list(evidence_dir.glob("design_baseline.*")) + list(evidence_dir.glob("design_baseline_v*.*"))
        implementation = list(evidence_dir.glob("implementation.*")) + list(evidence_dir.glob("implementation_v*.*"))
        baseline_required = self.baseline_required()
        summary_required = self.config.get("ui_design", {}).get("summary_required", True) and baseline_required
        summary_path = self.output_dir / "evidence" / "docs" / "ui_design_summary.md"
        missing = []
        if baseline_required and not baseline:
            missing.append("design_baseline.*")
        if not implementation:
            missing.append("implementation.*")
        if summary_required and not summary_path.exists():
            missing.append("docs/ui_design_summary.md")
        warnings: List[str] = []
        if not baseline_required and not baseline:
            warnings.append("design_baseline_missing")
        if not baseline_required and "docs/ui_design_summary.md" in missing:
            missing.remove("docs/ui_design_summary.md")
            warnings.append("ui_design_summary_missing")
        report = {
            "status": "passed" if not missing else "failed",
            "missing": missing,
            "path": str(evidence_dir),
            "warnings": warnings,
        }
        if not missing and baseline and implementation and self._ui_comparison_required():
            comparison = self._write_ui_comparison_report(evidence_dir, baseline[0], implementation[0])
            report["comparison_report"] = comparison
            if comparison.get("status") == "failed":
                if self._ui_comparison_required():
                    report["status"] = "failed"
                else:
                    report.setdefault("warnings", []).append("comparison_failed")
        return report

    def prepare_assets(self, requirements: Any) -> Dict[str, Any]:
        ui_cfg = self.config.get("ui_design", {})
        if ui_cfg.get("enabled", True) is False:
            return {"status": "skipped", "reason": "disabled"}

        if not self.requires_ui_baseline_from_requirements(requirements):
            return {"status": "skipped", "reason": "not_required"}

        evidence_dir = self.output_dir / "evidence" / "ui"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        baseline_candidates = list(evidence_dir.glob("design_baseline.*")) + list(
            evidence_dir.glob("design_baseline_v*.*")
        )
        user_baseline = self._resolve_user_baseline(evidence_dir)
        if user_baseline is not None:
            baseline = user_baseline
            baseline_source = "user"
        else:
            baseline = baseline_candidates[0] if baseline_candidates else None
            baseline_source = "existing" if baseline else "none"
        report: Dict[str, Any] = {"status": "passed", "baseline": str(baseline) if baseline else None}

        if baseline is None:
            output_name = ui_cfg.get("baseline_name", "design_baseline.png")
            output_path = evidence_dir / output_name
            prompt = self._build_ui_design_prompt(requirements)
            gen_result = self._run_nanobanna(prompt, output_path)
            if gen_result.get("status") != "completed":
                if self.allow_missing_baseline():
                    report.update(
                        {
                            "status": "skipped",
                            "reason": "baseline_unavailable",
                            "detail": gen_result,
                        }
                    )
                    self.memory.global_context["ui_design_source"] = "none"
                    return report
                report.update({"status": "failed", "reason": "baseline_generation_failed", "detail": gen_result})
                return report
            baseline = output_path
            baseline_source = "generated"
            report["baseline"] = str(baseline)
        report["baseline_source"] = baseline_source

        summary_path = self.output_dir / "evidence" / "docs" / "ui_design_summary.md"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if not summary_path.exists() or summary_path.stat().st_size == 0:
            summary_result = self._summarize_ui_baseline(baseline, requirements)
            report["summary"] = summary_result
            if summary_result.get("status") == "completed":
                summary_text = summary_result.get("summary", "")
                summary_path.write_text(summary_text, encoding="utf-8")
                self.memory.global_context["ui_design_summary"] = self._truncate(summary_text, 2000)
            elif ui_cfg.get("summary_required", True) and not self.allow_missing_baseline():
                report["status"] = "failed"
                report["reason"] = "summary_generation_failed"
                return report
        else:
            existing = summary_path.read_text(encoding="utf-8", errors="ignore")
            self.memory.global_context["ui_design_summary"] = self._truncate(existing, 2000)

        self.memory.global_context["ui_design"] = {
            "tool": ui_cfg.get("tool", "nanobanna"),
            "baseline": str(baseline),
            "summary_path": str(summary_path),
        }
        self.memory.global_context["ui_design_source"] = baseline_source
        return report

    def _should_force_ui_design(self) -> bool:
        ui_cfg = self.config.get("ui_design", {})
        if ui_cfg.get("force_if_no_frontend", True) is False:
            return False
        return not self._detect_frontend_presence(self.output_dir)

    def _detect_frontend_presence(self, root: Path, max_files: int = 300) -> bool:
        if self._frontend_detected is not None:
            return self._frontend_detected
        frontend_dirs = {"ui", "frontend", "web", "client", "app"}
        frontend_exts = {".html", ".css", ".js", ".jsx", ".tsx", ".vue", ".svelte"}
        ui_py_markers = ("streamlit", "gradio", "dash", "fastapi.templating", "jinja2")
        skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "output", "data", "evidence"}
        scanned = 0

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in skip_dirs]
            if Path(dirpath).name in frontend_dirs and filenames:
                self._frontend_detected = True
                return True
            for filename in filenames:
                scanned += 1
                if scanned > max_files:
                    break
                path = Path(dirpath) / filename
                if path.suffix in frontend_exts:
                    self._frontend_detected = True
                    return True
                if path.suffix == ".py":
                    try:
                        content = path.read_text(encoding="utf-8", errors="ignore")[:2000].lower()
                    except Exception:
                        continue
                    if any(marker in content for marker in ui_py_markers):
                        self._frontend_detected = True
                        return True
            if scanned > max_files:
                break
        self._frontend_detected = False
        return False

    def _ui_comparison_required(self) -> bool:
        if self.baseline_required():
            return True
        return self.config.get("ui_design", {}).get("comparison_required", False)

    def _ui_diff_threshold(self) -> float:
        raw = self.config.get("ui_design", {}).get("pixel_diff_threshold", 0.15)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.15

    def _ui_layout_threshold(self) -> float:
        raw = self.config.get("ui_design", {}).get("layout_similarity_threshold", 0.75)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.75

    def _ui_edge_threshold(self) -> int:
        raw = self.config.get("ui_design", {}).get("edge_threshold", 20)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 20

    def _ui_layout_compare_size(self) -> tuple[int, int]:
        raw = self.config.get("ui_design", {}).get("layout_compare_size", [192, 192])
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                return int(raw[0]), int(raw[1])
            except (TypeError, ValueError):
                pass
        return (192, 192)

    def _write_ui_comparison_report(self, evidence_dir: Path, baseline: Path, implementation: Path) -> Dict[str, Any]:
        comparison_required = self._ui_comparison_required()
        diff_threshold = self._ui_diff_threshold()
        warnings: List[str] = []

        def _digest(path: Path) -> str:
            hash_obj = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()

        report: Dict[str, Any] = {
            "status": "passed",
            "baseline": {
                "path": str(baseline),
                "size_bytes": baseline.stat().st_size,
                "sha256": _digest(baseline),
            },
            "implementation": {
                "path": str(implementation),
                "size_bytes": implementation.stat().st_size,
                "sha256": _digest(implementation),
            },
        }
        try:
            from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

            base_img = Image.open(baseline).convert("RGB")
            impl_img = Image.open(implementation).convert("RGB")
            report["sizes"] = {"baseline": base_img.size, "implementation": impl_img.size}

            layout_size = self._ui_layout_compare_size()
            edge_threshold = self._ui_edge_threshold()
            layout_threshold = self._ui_layout_threshold()

            def _edge_map(image: Image.Image) -> Image.Image:
                gray = image.convert("L")
                edges = gray.filter(ImageFilter.FIND_EDGES)
                edges = ImageOps.autocontrast(edges)
                return edges.point(lambda p: 255 if p > edge_threshold else 0)

            def _similarity(img_a: Image.Image, img_b: Image.Image) -> float:
                diff = ImageChops.difference(img_a, img_b)
                stat = ImageStat.Stat(diff)
                mean = stat.mean[0] if stat.mean else 255.0
                score = 1.0 - (mean / 255.0)
                return max(0.0, min(1.0, score))

            base_layout = _edge_map(base_img.resize(layout_size, Image.BILINEAR))
            impl_layout = _edge_map(impl_img.resize(layout_size, Image.BILINEAR))
            layout_similarity = _similarity(base_layout, impl_layout)
            report["layout_similarity"] = {
                "score": round(layout_similarity, 4),
                "threshold": layout_threshold,
                "status": "passed" if layout_similarity >= layout_threshold else "failed",
            }
            if layout_similarity < layout_threshold:
                if comparison_required:
                    report["status"] = "failed"
                else:
                    warnings.append("layout_similarity_below_threshold")

            if base_img.size != impl_img.size:
                if comparison_required:
                    report["status"] = "failed"
                else:
                    warnings.append("size_mismatch")
                report["pixel_diff"] = {"status": "failed", "reason": "size_mismatch"}
            else:
                diff = ImageChops.difference(base_img, impl_img)
                total_pixels = base_img.size[0] * base_img.size[1]
                diff_l = diff.convert("L")
                histogram = diff_l.histogram()
                mismatched = total_pixels - histogram[0]
                ratio = mismatched / total_pixels if total_pixels else 0.0
                diff_path = evidence_dir / "diff.png"
                diff.save(diff_path)
                threshold = diff_threshold
                report["pixel_diff"] = {
                    "status": "failed" if ratio > threshold else "passed",
                    "mismatch_ratio": ratio,
                    "threshold": threshold,
                    "diff_path": str(diff_path),
                }
                if ratio > threshold:
                    if comparison_required:
                        report["status"] = "failed"
                    else:
                        warnings.append("pixel_diff_exceeds_threshold")
        except Exception as exc:
            report["pixel_diff"] = {"status": "skipped", "reason": str(exc)}

        report_path = evidence_dir / "comparison.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["report_path"] = str(report_path)
        if warnings:
            report["warnings"] = warnings
        return report

    def _build_ui_design_prompt(self, requirements: Dict[str, Any]) -> str:
        goal = requirements.get("goal") or requirements.get("summary") or "未提供"
        features = requirements.get("key_features") or requirements.get("functional_requirements") or []
        if isinstance(features, list):
            features_text = "; ".join([str(item) for item in features if str(item).strip()])
        else:
            features_text = str(features)
        constraints = requirements.get("constraints") or []
        if isinstance(constraints, list):
            constraints_text = "; ".join([str(item) for item in constraints if str(item).strip()])
        else:
            constraints_text = str(constraints)
        prompt = (
            "为以下产品生成前端场景设计图（包含布局、关键组件、状态与交互提示）。"
            "要求：商业可用、结构清晰、信息层级明确。\n"
            f"目标: {goal}\n"
            f"关键功能: {features_text}\n"
            f"约束: {constraints_text}\n"
        )
        return prompt

    def _build_nanobanna_command(self, prompt: str, output_path: Path) -> List[str]:
        ui_cfg = self.config.get("ui_design", {})
        if isinstance(ui_cfg.get("command"), list):
            cmd = list(ui_cfg["command"])
            base_dir = find_project_root(Path(__file__).parent)
            for idx, arg in enumerate(cmd):
                if not isinstance(arg, str) or not arg.endswith(".py"):
                    continue
                script_path = Path(arg)
                if script_path.is_absolute():
                    continue
                cmd[idx] = str((base_dir / script_path).resolve())
            prompt_arg = ui_cfg.get("prompt_arg", "--prompt")
            output_arg = ui_cfg.get("output_arg", "--output")
            cmd.extend([prompt_arg, prompt, output_arg, str(output_path)])
            return cmd
        template = ui_cfg.get("command_template")
        if template:
            rendered = template.format(prompt=prompt, output=str(output_path))
            return shlex.split(rendered)
        return ["nanobanna", "--prompt", prompt, "--output", str(output_path)]

    def _run_nanobanna(self, prompt: str, output_path: Path) -> Dict[str, Any]:
        ui_cfg = self.config.get("ui_design", {})
        if ui_cfg.get("use_internal", True):
            try:
                from agents.dev_team.tools.nanobanna import generate_image

                model = ui_cfg.get("model", "gemini-2.5-flash-image")
                api_key = ui_cfg.get("api_key")
                return generate_image(prompt, output_path, model=model, api_key=api_key)
            except Exception as exc:
                return {"status": "failed", "reason": f"internal_error: {exc}"}

        cmd = self._build_nanobanna_command(prompt, output_path)
        if not cmd:
            return {"status": "failed", "reason": "command_missing"}
        if shutil.which(cmd[0]) is None:
            return {"status": "failed", "reason": f"{cmd[0]}_not_found"}
        timeout = ui_cfg.get("timeout", 60)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.output_dir,
            )
            if result.returncode != 0:
                return {
                    "status": "failed",
                    "reason": "command_failed",
                    "stderr": result.stderr[-2000:],
                    "stdout": result.stdout[-1000:],
                }
            if not output_path.exists():
                return {"status": "failed", "reason": "output_missing"}
            return {"status": "completed", "path": str(output_path)}
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}

    def _resolve_user_baseline(self, evidence_dir: Path) -> Optional[Path]:
        ui_cfg = self.config.get("ui_design", {})
        raw_path = ui_cfg.get("baseline_path") or os.environ.get("UI_DESIGN_BASELINE")
        if not raw_path:
            return None
        base_dir = find_project_root(Path(__file__).parent)
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        if not candidate.exists():
            return None
        suffix = candidate.suffix or ".png"
        target = evidence_dir / f"design_baseline{suffix}"
        if target.resolve() != candidate.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
        return target

    @staticmethod
    def _encode_image(path: Path, max_bytes: int = 2_000_000) -> Optional[str]:
        if not path.exists() or not path.is_file():
            return None
        if path.stat().st_size > max_bytes:
            return None
        mime, _ = mimetypes.guess_type(path.name)
        if not mime:
            mime = "image/png"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    def _summarize_ui_baseline(self, baseline: Path, requirements: Dict[str, Any]) -> Dict[str, Any]:
        ui_cfg = self.config.get("ui_design", {})
        if ui_cfg.get("summary_enabled", True) is False:
            return {"status": "skipped"}
        llm_cfg = self.config.get("llm", {})
        api_key = ui_cfg.get("summary_api_key") or llm_cfg.get("api_key")
        if not api_key:
            return {"status": "failed", "reason": "llm_not_configured"}
        model = ui_cfg.get("summary_model") or llm_cfg.get("model", "gpt-4o")
        api_base = ui_cfg.get("summary_api_base") or llm_cfg.get("api_base")
        encoded = self._encode_image(baseline)
        if not encoded:
            return {"status": "failed", "reason": "image_unavailable"}

        goal = requirements.get("goal") or requirements.get("summary") or ""
        prompt = (
            "你是资深前端设计评审官。请阅读设计基线图，输出可直接实现的前端设计摘要。\n"
            "必须包含：信息架构/布局、关键组件清单、交互与状态、视觉风格提示、可访问性/响应式注意事项。\n"
            "输出为中文 Markdown，避免空话。\n"
            f"目标: {goal}\n"
        )
        try:
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=api_base,
                temperature=0.2,
                timeout=60,
                max_retries=1,
            )
            messages = [
                SystemMessage(content="你是严谨的前端设计总结助手。"),
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": encoded, "detail": "high"}},
                    ]
                ),
            ]
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            return {"status": "completed", "summary": str(content).strip()}
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}

    @staticmethod
    def _truncate(text: str, limit: int = 2000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[Truncated]..."
