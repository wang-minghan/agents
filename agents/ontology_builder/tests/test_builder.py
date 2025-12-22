import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.ontology_builder.agent import (
    DEFAULT_TEMPLATE_PATH,
    RELATION_DEFS,
    OntologyObject,
    RelationEdge,
    build_all,
    build_design_data,
    render_owl,
)


class OntologyBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_csv = Path("/home/minghan/project/agents/input/本体对象关系列表.csv")
        self.template = DEFAULT_TEMPLATE_PATH

    def test_design_coverage(self) -> None:
        objects = [
            OntologyObject(label="用户", code="User", en_label="user"),
            OntologyObject(label="活动", code="Campaign", en_label="campaign"),
            OntologyObject(label="渠道", code="Channel", en_label="channel"),
        ]
        edges = [
            RelationEdge(
                subject="User",
                predicate="relatesTo",
                object="Campaign",
                predicate_label=RELATION_DEFS["relatesTo"]["label"],
            ),
            RelationEdge(
                subject="Campaign",
                predicate="includes",
                object="Channel",
                predicate_label=RELATION_DEFS["includes"]["label"],
            ),
        ]
        scenarios = [
            {
                "id": "scenario_1",
                "title": "场景一",
                "question": "q1",
                "objects": [obj.label for obj in objects],
                "object_ids": [obj.code for obj in objects],
                "edges": [],
                "attributes": ["attr_1"],
            },
            {
                "id": "scenario_2",
                "title": "场景二",
                "question": "q2",
                "objects": [obj.label for obj in objects],
                "object_ids": [obj.code for obj in objects],
                "edges": [],
                "attributes": ["attr_1"],
            },
            {
                "id": "scenario_3",
                "title": "场景三",
                "question": "q3",
                "objects": [obj.label for obj in objects],
                "object_ids": [obj.code for obj in objects],
                "edges": [],
                "attributes": ["attr_1"],
            },
        ]
        data = build_design_data(self.sample_csv, objects, edges, [], scenarios, [], [])
        object_ids = {obj["id"] for obj in data["objects"]}
        covered = set()
        for scenario in data["scenarios"]:
            covered.update(scenario["object_ids"])
        coverage = len(covered) / max(len(object_ids), 1)
        self.assertGreaterEqual(coverage, 0.8)

    def test_render_owl(self) -> None:
        objects = [
            OntologyObject(label="用户", code="User", en_label="user"),
            OntologyObject(label="活动", code="Campaign", en_label="campaign"),
        ]
        edges = [
            RelationEdge(
                subject="User",
                predicate="relatesTo",
                object="Campaign",
                predicate_label=RELATION_DEFS["relatesTo"]["label"],
            )
        ]
        scenarios = [
            {
                "id": "scenario_1",
                "title": "场景一",
                "question": "q1",
                "objects": [obj.label for obj in objects],
                "object_ids": [obj.code for obj in objects],
                "edges": [],
                "attributes": ["attr_1"],
            }
        ]
        data = build_design_data(self.sample_csv, objects, edges, [], scenarios, [], [])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "ontology.owl"
            render_owl(self.template, data, out_path)
            content = out_path.read_text(encoding="utf-8")
        self.assertIn("本体属性", content)
        self.assertIn("暂无属性定义", content)
        sample_obj = data["objects"][0]["id"]
        self.assertIn(f"ex:{sample_obj} a owl:Class", content)

    def test_roundtable_ai_pipeline_with_mock_llm(self) -> None:
        def fake_invoke_llm(_llm, prompt: str) -> str:
            if "CSV预览" in prompt:
                return (
                    '{"source_type":"fields","tables":[{"table_name":"用户","table_en":"user",'
                    '"fields":[{"field_name":"用户ID","field_en":"user_id","data_type":"","comment":""}]}]}'
                )
            if "表与字段摘要" in prompt:
                return (
                    '{"objects":[{"table_name":"用户","object_name":"用户","object_en":"user"}],'
                    '"relations":[],"scenarios":[{"title":"规模与价值保有提升","question":"q","objects":["用户"],'
                    '"key_attributes":["用户ID"]},'
                    '{"title":"流失风险预警与挽留","question":"q","objects":["用户"],'
                    '"key_attributes":["用户ID"]},'
                    '{"title":"潜在客户挖掘与体验改善","question":"q","objects":["用户"],'
                    '"key_attributes":["用户ID"]}]}'
                )
            if "设计数据" in prompt and "规则审计问题" in prompt:
                return '{"status":"pass","issues":[]}'
            if "OWL内容" in prompt:
                return '{"status":"pass","issues":[]}'
            return "{}"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with patch("agents.ontology_builder.agent._build_llm", return_value=None), patch(
                "agents.ontology_builder.agent._invoke_llm", side_effect=fake_invoke_llm
            ):
                result = build_all(
                    input_csv=self.sample_csv,
                    output_dir=tmpdir_path,
                    max_rounds=1,
                )
            design_doc = Path(result["design_doc"]).read_text(encoding="utf-8")
            self.assertIn("关键属性：", design_doc)
            self.assertTrue(Path(result["owl"]).exists())


if __name__ == "__main__":
    unittest.main()
