from __future__ import annotations

import selectors
import time
from pathlib import Path
from subprocess import PIPE, Popen

import streamlit as st


def render_excel_to_csv(base_dir: Path) -> None:
    st.markdown('<div class="section-title">Excel to CSV</div>', unsafe_allow_html=True)
    st.caption("批量转换 Excel，并生成结构审核报告。")

    input_dir = st.text_input("输入目录", value=str(base_dir / "input"))
    output_dir = st.text_input("输出目录", value=str(base_dir / "output"))
    llm_config = st.text_input("LLM 配置路径", value=str(base_dir / "configs" / "llm.yaml"))

    if st.button("运行转换"):
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
