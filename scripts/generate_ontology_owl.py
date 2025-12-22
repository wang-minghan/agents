"""Generate OWL from template and design JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.ontology_builder.agent import DEFAULT_TEMPLATE_PATH, render_owl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OWL from design JSON")
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="OWL模板路径",
    )
    parser.add_argument(
        "--design",
        default="output/ontology_builder_design.json",
        help="设计数据JSON路径",
    )
    parser.add_argument(
        "--output",
        default="output/ontology.owl",
        help="OWL输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design_data = json.loads(Path(args.design).read_text(encoding="utf-8"))
    render_owl(Path(args.template), design_data, Path(args.output))


if __name__ == "__main__":
    main()
