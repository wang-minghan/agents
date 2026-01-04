from __future__ import annotations

import json
import queue
import threading
from contextlib import redirect_stdout
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from agents.dev_team.architect.agent import load_config as load_planner_config
from agents.dev_team.architect.agent import run_architect


def _parse_json(text: str, label: str) -> tuple[object | None, str | None]:
    if not text.strip():
        return None, None
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"{label} JSON 解析失败: {exc}"


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


def render_architect(base_dir: Path) -> None:
    if st.session_state.pop("architect_rerun", False):
        pass
    st.markdown('<div class="section-title">架构师</div>', unsafe_allow_html=True)
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

    st.markdown('<a id="planner-form"></a>', unsafe_allow_html=True)
    st.markdown(
        "<div class='flow-card'><div class='flow-kicker'>协作路径</div>"
        "<div class='flow-text'>输入需求 → 校验约束 → 规划角色 → 输出 JD</div></div>",
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([1.8, 1])
    with col_main:
        user_input = st.text_area(
            "需求描述",
            height=240,
            placeholder="例如：我想做一个支持高并发的秒杀系统，需要考虑缓存击穿、雪崩以及分布式锁。",
            key="architect_user_input",
        )
        constraints_text = st.text_area(
            "约束（JSON）",
            height=150,
            placeholder='例如：{"tech_stack":["python","fastapi"],"must_have":["redis","mysql"]}',
            key="architect_constraints",
        )

    with col_side:
        tab_basic, tab_snapshot, tab_resume = st.tabs(["迭代", "快照", "续跑"])
        with tab_basic:
            max_iterations = st.slider(
                "最大迭代轮次", min_value=1, max_value=6, value=3, key="architect_iters"
            )
            validation_threshold = st.slider(
                "验证阈值",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                key="architect_threshold",
            )
        with tab_snapshot:
            snapshot_enabled = st.checkbox("保存规划快照", value=True, key="architect_snapshot")
            snapshot_dir = st.text_input(
                "快照目录",
                value=str(base_dir / "agents" / "dev_team" / "architect" / "output" / "snapshots"),
                key="architect_snapshot_dir",
            )
        with tab_resume:
            user_feedback = st.text_area("用户补充反馈（可选）", height=120, key="architect_feedback")
            planner_state_text = st.text_area("planner_state（JSON，可选）", height=160, key="architect_state")

    run_col, hint_col = st.columns([1.1, 1.9])
    with run_col:
        run_clicked = st.button("运行 Task Planner", use_container_width=True)
    with hint_col:
        st.markdown(
            "<div class='run-hint'>建议先填需求与约束，再运行。若需要续跑，回填 planner_state。</div>",
            unsafe_allow_html=True,
        )

    if run_clicked:
        if not user_input.strip():
            st.error("请先填写需求描述。")
            return

        constraints, constraint_error = _parse_json(constraints_text, "constraints")
        if constraint_error:
            st.error(constraint_error)
            return

        planner_state, planner_state_error = _parse_json(planner_state_text, "planner_state")
        if planner_state_error:
            st.error(planner_state_error)
            return

        config = load_planner_config()
        workflow = config.setdefault("workflow", {})
        workflow["max_iterations"] = int(max_iterations)
        workflow["validation_threshold"] = float(validation_threshold)
        workflow["snapshot_enabled"] = bool(snapshot_enabled)
        workflow["snapshot_dir"] = snapshot_dir

        input_data = {"user_input": user_input}
        if constraints is not None:
            input_data["constraints"] = constraints
        if user_feedback.strip():
            input_data["user_feedback"] = user_feedback.strip()
        if planner_state is not None:
            input_data["planner_state"] = planner_state

        status_box = st.status("步骤 1/3：分析需求", expanded=True)
        progress = st.progress(0.1)
        log_box = st.empty()

        q: queue.Queue[str | None] = queue.Queue()

        def _runner() -> None:
            buffer = _QueueWriter(q)
            with redirect_stdout(buffer):
                result = run_architect(input_data, config)
            st.session_state["architect_result"] = result
            q.put(None)

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

        log_box.write_stream(_stream_queue(q))
        status_box.update(label="步骤 2/3：拆解与优化", state="running")
        progress.progress(0.7)
        status_box.update(label="步骤 3/3：完成", state="complete")
        progress.progress(1.0)
        st.session_state["architect_rerun"] = True
        st.session_state["architect_scroll"] = True
        st.rerun()

    result = st.session_state.get("architect_result")
    if not result:
        st.markdown('<div class="status-chip">尚未运行</div>', unsafe_allow_html=True)
        return

    status = result.get("status")
    st.markdown('<a id="planner-status"></a>', unsafe_allow_html=True)
    if status == "completed":
        st.success("规划完成")
    elif status == "needs_feedback":
        st.warning("需要补充信息")
        st.session_state["architect_scroll_status"] = True
    else:
        st.error(f"执行失败: {result.get('error')}")
        st.session_state["architect_scroll_status"] = True

    if status == "needs_feedback":
        planner_state_payload = result.get("planner_state")
        if planner_state_payload:
            st.markdown(
                "<div class='alert-card'><strong>需要补充信息</strong>："
                "已生成续跑状态，建议回填并补充反馈。</div>",
                unsafe_allow_html=True,
            )
            if st.button("回到输入区"):
                st.session_state["architect_scroll_form"] = True
            if st.button("将 planner_state 填回表单"):
                st.session_state["architect_state"] = json.dumps(
                    planner_state_payload, ensure_ascii=False, indent=2
                )
                st.info("已填回 planner_state，请补充反馈后再运行。")
    elif status not in ("completed", None):
        st.markdown(
            "<div class='alert-card'><strong>执行失败</strong>："
            "请查看错误详情并调整输入。</div>",
            unsafe_allow_html=True,
        )
        if st.button("回到输入区"):
            st.session_state["architect_scroll_form"] = True
        with st.expander("错误详情", expanded=False):
            st.code(str(result.get("error") or "Unknown error"), language="text")

    st.markdown('<a id="planner-result"></a>', unsafe_allow_html=True)
    st.subheader("结果概览")
    st.markdown('<div class="panel-card result-highlight">', unsafe_allow_html=True)
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.markdown(
            f"<div class='panel-card'><div class='panel-label'>状态</div><div class='panel-value'>{status}</div></div>",
            unsafe_allow_html=True,
        )
    with summary_col2:
        tasks_count = len(result.get("tasks") or [])
        st.markdown(
            f"<div class='panel-card'><div class='panel-label'>任务数</div><div class='panel-value'>{tasks_count}</div></div>",
            unsafe_allow_html=True,
        )
    with summary_col3:
        snapshot_path = result.get("snapshot_path") or "-"
        st.markdown(
            f"<div class='panel-card'><div class='panel-label'>快照</div><div class='panel-value'>{snapshot_path}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.pop("architect_scroll", False):
        components.html(
            """
            <script>
              const target = parent.document.getElementById("planner-result");
              if (target) {
                target.scrollIntoView({behavior: "smooth", block: "start"});
              }
            </script>
            """,
            height=0,
        )

    with st.expander("Requirements / Tasks / JDs", expanded=True):
        st.json(
            {
                "requirements": result.get("requirements"),
                "tasks": result.get("tasks"),
                "final_jds": result.get("final_jds"),
            }
        )

    download_payload = json.dumps(result, ensure_ascii=False, indent=2)
    st.download_button(
        "下载结果 JSON",
        data=download_payload,
        file_name="architect_result.json",
        mime="application/json",
    )

    if st.session_state.pop("architect_scroll_status", False):
        components.html(
            """
            <script>
              const target = parent.document.getElementById("planner-status");
              if (target) {
                target.scrollIntoView({behavior: "smooth", block: "start"});
              }
            </script>
            """,
            height=0,
        )

    if st.session_state.pop("architect_scroll_form", False):
        components.html(
            """
            <script>
              const target = parent.document.getElementById("planner-form");
              if (target) {
                target.scrollIntoView({behavior: "smooth", block: "start"});
              }
            </script>
            """,
            height=0,
        )
