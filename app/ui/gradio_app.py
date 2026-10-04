"""可选前端：Gradio 版界面。

启动：python -m app.ui.gradio_app
需要先安装：pip install gradio
"""

from __future__ import annotations

import time

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.services import get_services


def _documents_table() -> list[list]:
    with session_scope() as db:
        documents = repo.list_documents(db, limit=100)
        return [
            [
                document.name,
                document.status,
                document.chunk_count,
                f"{document.file_size / 1024:.1f} KB",
                document.created_at.strftime("%Y-%m-%d %H:%M") if document.created_at else "",
            ]
            for document in documents
        ]


def _memories_table(user_id: str) -> list[list]:
    return [
        [item["category"], item["content"], f"{item['importance']:.2f}"]
        for item in get_services().long_term.list(user_id)
    ]


def _wait(task_id: str, timeout: float = 300.0) -> str:
    services = get_services()
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = services.ingest.status(task_id)
        if status and status["status"] in {"succeeded", "failed", "skipped"}:
            return f"{status['status']}｜{status['name']}｜{status['message']}"
        time.sleep(0.5)
    return "任务仍在进行中，请稍后在「解析进度」中查看"


def build_ui():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("未安装 gradio，请先执行：pip install gradio") from exc

    settings = get_settings()
    services = get_services()

    def chat_fn(message: str, history: list, session_id: str | None):
        if not message.strip():
            return history, session_id
        if not session_id:
            with session_scope() as db:
                session_id = repo.create_session(db, user_id=settings.default_user_id).id
        result = services.executor.run(message, session_id=session_id)
        answer = result.answer
        if result.citations:
            answer += "\n\n**参考来源**\n" + "\n".join(
                f"- [{item['index']}] 《{item['document_name']}》"
                + (f"（{item['heading']}）" if item.get("heading") else "")
                for item in result.citations
            )
        answer += f"\n\n<sub>意图：{result.intent}｜依据度：{result.guardrails.get('grounded_ratio', 0):.0%}｜耗时 {result.latency_ms} ms</sub>"
        history = history + [[message, answer]]
        return history, session_id

    with gr.Blocks(title="个人知识库智能问答 Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 个人知识库智能问答 Agent\n只基于你上传的私有资料作答，每条结论都标注来源。")
        session_state = gr.State(None)

        with gr.Tab("对话"):
            chatbot = gr.Chatbot(height=520, type="tuples")
            with gr.Row():
                question = gr.Textbox(placeholder="基于知识库提问…", scale=4, show_label=False)
                send = gr.Button("发送", variant="primary", scale=1)
            clear = gr.Button("新建会话")
            send.click(chat_fn, [question, chatbot, session_state], [chatbot, session_state]).then(
                lambda: "", outputs=question
            )
            question.submit(chat_fn, [question, chatbot, session_state], [chatbot, session_state]).then(
                lambda: "", outputs=question
            )
            clear.click(lambda: ([], None), outputs=[chatbot, session_state])

        with gr.Tab("知识库管理"):
            upload = gr.File(file_count="multiple", label="上传文档（TXT / MD / PDF / DOCX / HTML）")
            ingest_btn = gr.Button("开始导入", variant="primary")
            progress = gr.Textbox(label="导入结果", lines=4)
            url = gr.Textbox(label="或导入网页链接")
            url_btn = gr.Button("导入网页")

            def ingest_files(files):
                if not files:
                    return "请先选择文件"
                messages = []
                for file in files:
                    path = file.name if hasattr(file, "name") else str(file)
                    task_id = services.ingest.submit_path(path, name=None)
                    messages.append(_wait(task_id))
                return "\n".join(messages)

            def ingest_url(url_value: str):
                if not url_value.strip():
                    return "请填写网页链接"
                return _wait(services.ingest.submit_url(url_value.strip()))

            ingest_btn.click(ingest_files, upload, progress)
            url_btn.click(ingest_url, url, progress)
            refresh = gr.Button("刷新文档列表")
            table = gr.Dataframe(
                headers=["文档", "状态", "片段数", "大小", "上传时间"], value=_documents_table(), interactive=False
            )
            refresh.click(lambda: _documents_table(), outputs=table)

        with gr.Tab("长期记忆"):
            memory_refresh = gr.Button("刷新")
            memory_table = gr.Dataframe(
                headers=["类型", "内容", "重要度"],
                value=_memories_table(settings.default_user_id),
                interactive=False,
            )
            memory_refresh.click(lambda: _memories_table(settings.default_user_id), outputs=memory_table)

    return demo


def main() -> None:
    build_ui().launch(server_name=get_settings().api_host, server_port=get_settings().api_port + 1)


if __name__ == "__main__":
    main()
