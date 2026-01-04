from __future__ import annotations

import json
import queue
import threading
from contextlib import redirect_stdout
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from agents.dev_team.orchestrator import Orchestrator
from agents.dev_team.utils import load_config as load_dev_config
from agents.task_planner.agent import build_agent as build_planner


class _QueueWriter:
    def __init__(self, q: queue.Queue[str | None]) -> None:
        self._queue = q

    def write(self, text: str) -> None:
        if text:
            self._queue.put(text)

    def flush(self) -> None:
        return None


def _stream_queue(q: queue.Queue[str | None]):
    while True:
        item = q.get()
        if item is None:
            break
        yield item


def render_dev_team(base_dir: Path) -> None:
    if st.session_state.pop("dev_team_rerun", False):
        pass
    st.markdown('<div class="section-title">Dev Team</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <style>
          .panel-card {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: 24px;
            padding: 1rem 1.2rem;
            box-shadow: 0 18px 45px rgba(16, 15, 20, 0.12);
          }
          .flow-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(0, 0, 0, 0.06);
            border-radius: 24px;
            padding: 0.85rem 1.2rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 16px 32px rgba(16, 15, 20, 0.08);
          }
          .flow-kicker {
            text-transform: uppercase;
            letter-spacing: 0.16rem;
            font-size: 0.65rem;
            color: #6d6a77;
          }
          .flow-text {
            margin-top: 0.3rem;
            font-size: 1rem;
            font-weight: 600;
            color: #1b1b1f;
          }
          .panel-label {
            font-size: 0.7rem;
            letter-spacing: 0.14rem;
            text-transform: uppercase;
            color: #6d6a77;
          }
          .panel-value {
            font-size: 1.05rem;
            font-weight: 600;
            color: #1b1b1f;
            margin-top: 0.3rem;
          }
          .result-highlight {
            border-color: rgba(255, 111, 61, 0.45);
            box-shadow: 0 16px 32px rgba(255, 111, 61, 0.12);
          }
          .alert-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(255, 111, 61, 0.35);
            border-radius: 16px;
            padding: 0.9rem 1.1rem;
            box-shadow: 0 12px 22px rgba(255, 111, 61, 0.12);
            margin: 0.6rem 0 1rem;
          }
          div[data-testid="stTabs"] > div {
            background: transparent;
          }
          div[data-testid="stTabs"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: 24px;
            padding: 0.9rem 1rem 1rem;
            box-shadow: 0 18px 45px rgba(16, 15, 20, 0.12);
          }
          div[data-baseweb="tab-list"] {
            padding-top: 0;
            margin-top: -0.2rem;
          }
          .run-hint {
            color: #6d6a77;
            font-size: 0.85rem;
            margin-top: 0.35rem;
          }
          .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(0, 0, 0, 0.08);
            color: #6d6a77;
            font-size: 0.8rem;
            margin-top: 0.6rem;
          }
          div[data-testid="stTextArea"] textarea,
          div[data-testid="stTextInput"] input {
            color: #f6f1e9;
            background-color: #24252b;
            border: 1px solid rgba(255, 255, 255, 0.12);
          }
          div[data-testid="stTextArea"] textarea::placeholder,
          div[data-testid="stTextInput"] input::placeholder {
            color: #c7c0b6;
          }
          div[data-testid="stTextArea"] label,
          div[data-testid="stTextInput"] label {
            color: #3e3a44;
            font-weight: 600;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<a id="dev-form"></a>', unsafe_allow_html=True)
    st.markdown(
        "<div class='flow-card'><div class='flow-kicker'>协作路径</div>"
        "<div class='flow-text'>输入需求 → 规划角色 → 团队协作 → QA 校验 → 报告归档</div></div>",
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([1.8, 1])
    with col_main:
        user_input = st.text_area(
            "需求描述",
            height=240,
            placeholder="例如：构建一个支持团队协作的任务看板，需含权限、审计日志与导出。",
        )
    with col_side:
        tab_flow, tab_exec, tab_output = st.tabs(["协作流程", "执行策略", "输出"])
        with tab_flow:
            max_rounds = st.slider("最大协作轮次", min_value=1, max_value=6, value=3)
            force_qa_on_success = st.checkbox("测试通过后仍执行 QA 审查", value=False)
            post_success_qa_rounds = st.slider("测试通过后的 QA 轮次", min_value=0, max_value=2, value=0)
        with tab_exec:
            allow_unsafe_execution = st.checkbox("允许本地执行测试（危险）", value=False)
        with tab_output:
            output_dir = st.text_input(
                "输出目录",
                value=str(base_dir / "agents" / "dev_team" / "output" / "codebase"),
            )

    run_col, hint_col = st.columns([1.1, 1.9])
    with run_col:
        run_clicked = st.button("运行 Dev Team", use_container_width=True)
    with hint_col:
        st.markdown(
            "<div class='run-hint'>运行将调用 Planner 与协作流程，请确保 LLM 配置有效。</div>",
            unsafe_allow_html=True,
        )

    if run_clicked:
        if not user_input.strip():
            st.error("请先填写需求描述。")
            return

        config = load_dev_config()
        config["output_dir"] = output_dir
        config["allow_unsafe_execution"] = bool(allow_unsafe_execution)
        config["collaboration"] = {
            "force_qa_on_success": bool(force_qa_on_success),
            "post_success_qa_rounds": int(post_success_qa_rounds),
        }

        planner = build_planner()
        status_box = st.status("步骤 1/3：规划角色", expanded=True)
        progress = st.progress(0.1)
        log_box = st.empty()
        q: queue.Queue[str | None] = queue.Queue()

        def _runner() -> None:
            buffer = _QueueWriter(q)
            result: dict
            with redirect_stdout(buffer):
                planner_result = planner.invoke({"user_input": user_input})
                planner_status = planner_result.get("status")
                if planner_status == "completed":
                    orchestrator = Orchestrator(config, output_dir=output_dir)
                    orchestrator.initialize_team(planner_result)
                    result = orchestrator.run_collaboration(max_rounds=max_rounds)
                elif planner_status == "needs_feedback":
                    result = {
                        "status": "needs_feedback",
                        "planner_result": planner_result,
                    }
                else:
                    result = {
                        "status": "error",
                        "error": planner_result.get("error", "planner_failed"),
                        "planner_result": planner_result,
                    }
            st.session_state["dev_team_result"] = result
            q.put(None)

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

        log_box.write_stream(_stream_queue(q))
        status_box.update(label="步骤 2/3：协作执行", state="running")
        progress.progress(0.7)
        status_box.update(label="步骤 3/3：完成", state="complete")
        progress.progress(1.0)
        st.session_state["dev_team_scroll"] = True
        st.session_state["dev_team_rerun"] = True
        st.rerun()

    result = st.session_state.get("dev_team_result")
    if not result:
        st.markdown('<div class="status-chip">尚未运行</div>', unsafe_allow_html=True)
        return

    status = result.get("status")
    st.markdown('<a id="dev-status"></a>', unsafe_allow_html=True)
    if status == "passed":
        st.success("协作完成（测试通过）")
    elif status == "needs_feedback":
        st.warning("Planner 需要补充信息，先补全再运行协作。")
        st.session_state["dev_team_scroll_status"] = True
        st.markdown(
            "<div class='alert-card'><strong>需要补充信息</strong>："
            "请先补全 Planner 输出再运行协作。</div>",
            unsafe_allow_html=True,
        )
        if st.button("回到输入区"):
            st.session_state["dev_team_scroll_form"] = True
    elif status == "error":
        st.error(f"执行失败: {result.get('error')}")
        st.session_state["dev_team_scroll_status"] = True
        st.markdown(
            "<div class='alert-card'><strong>执行失败</strong>："
            "请查看错误详情并调整输入。</div>",
            unsafe_allow_html=True,
        )
        if st.button("回到输入区"):
            st.session_state["dev_team_scroll_form"] = True
        with st.expander("错误详情", expanded=False):
            st.code(str(result.get("error") or "Unknown error"), language="text")
    else:
        st.info(f"协作完成：{status}")

    st.markdown('<a id="dev-result"></a>', unsafe_allow_html=True)
    st.subheader("结果概览")
    st.markdown('<div class="panel-card result-highlight">', unsafe_allow_html=True)
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.markdown(
            f"<div class='panel-card'><div class='panel-label'>状态</div><div class='panel-value'>{status}</div></div>",
            unsafe_allow_html=True,
        )
    with summary_col2:
        report_path = (result.get("report") or {}).get("report_path") or "-"
        st.markdown(
            f"<div class='panel-card'><div class='panel-label'>报告</div><div class='panel-value'>{report_path}</div></div>",
            unsafe_allow_html=True,
        )
    with summary_col3:
        saved_count = len((result.get("report") or {}).get("saved_files") or [])
        st.markdown(
            f"<div class='panel-card'><div class='panel-label'>文件数</div><div class='panel-value'>{saved_count}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.pop("dev_team_scroll", False):
        components.html(
            """
            <script>
              const target = parent.document.getElementById("dev-result");
              if (target) {
                target.scrollIntoView({behavior: "smooth", block: "start"});
              }
            </script>
            """,
            height=0,
        )

    with st.expander("产出与报告", expanded=True):
        st.json(result)

    logs = st.session_state.get("dev_team_logs")
    if logs:
        with st.expander("运行日志"):
            st.code(logs, language="text")

    download_payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    st.download_button(
        "下载结果 JSON",
        data=download_payload,
        file_name="dev_team_result.json",
        mime="application/json",
    )

    if st.session_state.pop("dev_team_scroll_status", False):
        components.html(
            """
            <script>
              const target = parent.document.getElementById("dev-status");
              if (target) {
                target.scrollIntoView({behavior: "smooth", block: "start"});
              }
            </script>
            """,
            height=0,
        )

    if st.session_state.pop("dev_team_scroll_form", False):
        components.html(
            """
            <script>
              const target = parent.document.getElementById("dev-form");
              if (target) {
                target.scrollIntoView({behavior: "smooth", block: "start"});
              }
            </script>
            """,
            height=0,
        )
