from __future__ import annotations

import selectors
import time
from pathlib import Path
from subprocess import PIPE, Popen

import streamlit as st


def render_excel_to_csv(base_dir: Path) -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&family=Work+Sans:wght@300;400;600&display=swap');
          :root {
            --ink: #1a1a1f;
            --muted: #6b6b75;
            --glass: rgba(255, 255, 255, 0.88);
            --sun: #f5b03e;
            --ocean: #2e6f9e;
            --plum: #5b3b58;
          }
          .excel-hero {
            background: linear-gradient(135deg, rgba(245,176,62,0.25), rgba(46,111,158,0.2)),
              radial-gradient(circle at top right, rgba(91,59,88,0.18), transparent 55%);
            border-radius: 28px;
            padding: 1.8rem 2rem;
            border: 1px solid rgba(0, 0, 0, 0.08);
            box-shadow: 0 24px 55px rgba(24, 24, 33, 0.18);
          }
          .excel-title {
            font-family: 'Fraunces', serif;
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--ink);
          }
          .excel-subtitle {
            font-family: 'Work Sans', sans-serif;
            color: var(--muted);
            margin-top: 0.6rem;
            font-size: 1rem;
          }
          .excel-card {
            background: var(--glass);
            border-radius: 22px;
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(0, 0, 0, 0.08);
            box-shadow: 0 18px 42px rgba(24, 24, 33, 0.12);
          }
          .excel-label {
            font-family: 'Work Sans', sans-serif;
            font-size: 0.72rem;
            letter-spacing: 0.18rem;
            text-transform: uppercase;
            color: var(--muted);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="excel-hero">
          <div class="excel-title">Excel → CSV Studio</div>
          <div class="excel-subtitle">批量转换、结构审核、生成可追溯审计报告。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='excel-label'>配置</div>", unsafe_allow_html=True)
    with st.container():
        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            st.markdown("<div class='excel-card'>", unsafe_allow_html=True)
            input_dir = st.text_input("输入目录", value=str(base_dir / "input"))
            output_dir = st.text_input("输出目录", value=str(base_dir / "output"))
            st.markdown("</div>", unsafe_allow_html=True)
        with col_right:
            st.markdown("<div class='excel-card'>", unsafe_allow_html=True)
            llm_config = st.text_input("LLM 配置路径", value=str(base_dir / "configs" / "llm.yaml"))
            st.markdown("建议：将输入按业务主题分目录，便于审计归档。")
            st.markdown("</div>", unsafe_allow_html=True)

    run_col, info_col = st.columns([1, 1.5])
    with run_col:
        run_clicked = st.button("运行转换", use_container_width=True)
    with info_col:
        st.info("执行后自动生成 __audit.md 报告，可直接复用为交付证据。")

    if run_clicked:
        python_bin = base_dir / ".venv" / "bin" / "python"
        cmd = [
            str(python_bin),
            "agents/excel_to_csv/agent.py",
            "--input",
            input_dir,
            "--output",
            output_dir,
            "--llm-config",
            llm_config,
        ]
        st.subheader("运行过程")
        status_box = st.empty()
        log_box = st.empty()
        err_box = st.empty()
        start = time.time()
        logs: list[str] = []
        errors: list[str] = []
        status_box.info("任务启动中...")
        with Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, bufsize=1) as proc:
            selector = selectors.DefaultSelector()
            if proc.stdout is not None:
                selector.register(proc.stdout, selectors.EVENT_READ)
            if proc.stderr is not None:
                selector.register(proc.stderr, selectors.EVENT_READ)
            while True:
                for key, _ in selector.select(timeout=0.1):
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    if key.fileobj is proc.stdout:
                        logs.append(line.rstrip())
                    else:
                        errors.append(line.rstrip())
                elapsed = int(time.time() - start)
                status_box.info(f"运行中... {elapsed}s")
                if logs:
                    log_box.code("\n".join(logs[-200:]), language="text")
                if errors:
                    err_box.error("\n".join(errors[-50:]))
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
        if proc.returncode == 0:
            status_box.success("已完成")
        else:
            status_box.error(f"失败，退出码: {proc.returncode}")
        if not logs:
            log_box.code("完成", language="text")

    st.divider()
    st.subheader("最近的审核报告")
    output_path = Path(output_dir)
    if output_path.exists():
        audits = sorted(
            output_path.rglob("*__audit.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for audit in audits[:10]:
            with st.expander(str(audit)):
                st.code(audit.read_text(encoding="utf-8"), language="markdown")
    else:
        st.info("未找到输出目录。")
