# Ontology Builder Agent

基于CSV元数据（最坏情况仅字段名）构建本体对象、关系设计、设计文档与OWL，流程全程由AI完成。

## 使用

```bash
python agents/ontology_builder/agent.py \
  --input input/本体对象关系列表.csv \
  --output output
```

如需指定LLM配置：

```bash
python agents/ontology_builder/agent.py \
  --input input/本体对象关系列表.csv \
  --output output \
  --llm-config configs/llm.yaml
```

模板文件为内置资源，不暴露为输入参数。

## 输出（默认）

- output/ontology_builder_design.md
- output/ontology.owl

## 工具列表

- agents/ontology_builder/tools/README.md

## 运行态流程

1) AI1 解析 CSV 原始内容并识别表/关系，输出 raw_meta  
2) AI2 设计对象/关系/场景（场景标题由AI推断），输出设计文档与映射  
3) AI3 审计设计，失败则回流  
4) AI4 生成 OWL  
5) AI5 抽检 OWL，失败则回流并生成修订 CSV

## OWL 生成脚本

```bash
python scripts/generate_ontology_owl.py \
  --design output/ontology_builder_design.json \
  --output output/ontology.owl
```
