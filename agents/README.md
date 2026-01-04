# Agents Directory

This directory contains individual AI agents, each designed for a specific purpose.

## List of Agents

| Agent Name | Description | Status |
| :--- | :--- | :--- |
| **excel_to_csv** | Converts Excel files to CSV format | Active |
| **restaurant_recommender** | Recommends restaurants based on location, price, and history | Active |
| **task_planner** | Breaks down requirements into tasks and JDs, with validation loops, output checks, constraints injection, and optional planning snapshots. | Active |
| **dev_team** | Orchestrates multi-agent engineering + QA collaboration with shared memory, test gating, and structured collaboration reports. | Active |

## Creating a New Agent

1. Create a new directory: `agents/<new_agent_name>/`
2. Add `agent.py`, `prompts/`, and `config/`.
3. Implement `build_agent()` in `agent.py`.
4. Add documentation here.
