import torch

from utils.chatglm import models as chatglm_models
from .chain import chain_ui
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"


def update_status(history, status):
    history = history + [[None, status]]
    print(status)
    return history


def chatglm_ui():
    with gr.Tab('Chat'):
        def revise(history, latest_message):
            history[-1] = (history[-1][0], latest_message)
            return history, ''

        def revoke(history):
            if len(history) >= 1:
                history.pop()
            return history

        def interrupt(allow_generate):
            allow_generate[0] = False

        def reset_state():
            return [], []

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=800)
            with gr.Column(scale=1):
                with gr.Row():
                    max_length = gr.Slider(32, 4096, value=2048, step=1, label="Maximum length", interactive=True)
                    top_p = gr.Slider(0.01, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                    temperature = gr.Slider(0.01, 5, value=0.95, step=0.01, label="Temperature", interactive=True)

                with gr.Row():
                    query = gr.Textbox(show_label=False, placeholder="Prompts", lines=4).style(container=False)
                    generate_button = gr.Button("生成")
                with gr.Row():
                    continue_message = gr.Textbox(
                        show_label=False, placeholder="Continue message", lines=2).style(container=False)
                    continue_btn = gr.Button("续写")
                    revise_message = gr.Textbox(
                        show_label=False, placeholder="Revise message", lines=2).style(container=False)
                    revise_btn = gr.Button("修订")
                    revoke_btn = gr.Button("撤回")
                    interrupt_btn = gr.Button("终止生成")
                    reset_btn = gr.Button("清空")

                with gr.Row():
                    chat_mode = gr.Radio(
                        ["ChatGLM 对话", "知识库问答"],
                        label="请选择使用模式",
                        value="知识库问答",
                    )
                    chain_ui(mode=chat_mode, chatbot=chatbot, query=query)

        history = gr.State([])
        allow_generate = gr.State([True])
        blank_input = gr.State("")
        reset_btn.click(reset_state, outputs=[chatbot, history], show_progress=True)
        generate_button.click(
            chatglm_models.chatglm.predict_continue,
            inputs=[query, blank_input, max_length, top_p, temperature, allow_generate, history],
            outputs=[chatbot, query]
        )
        revise_btn.click(revise, inputs=[history, revise_message], outputs=[chatbot, revise_message])
        revoke_btn.click(revoke, inputs=[history], outputs=[chatbot])
        continue_btn.click(
            chatglm_models.chatglm.predict_continue,
            inputs=[query, continue_message, max_length, top_p, temperature, allow_generate, history],
            outputs=[chatbot, query, continue_message]
        )
        interrupt_btn.click(interrupt, inputs=[allow_generate])
