import torch

from utils.chatglm import models as chatglm_models
from .chain import chain_ui
import gradio as gr
from .chain import get_answer

device = "cuda" if torch.cuda.is_available() else "cpu"


def update_status(history, status):
    history = history + [[None, status]]
    print(status)
    return history


def predict_continue(
        query,
        latest_message,
        max_length,
        top_p,
        temperature,
        allow_generate,
        history,
        chat_mode='chat',

):
    if chat_mode == "chat":
        yield chatglm_models.chatglm.predict_continue(
            query=query,
            latest_message=latest_message,
            max_lengt=max_length,
            top_p=top_p,
            temperature=temperature,
            allow_generate=allow_generate,
            history=history,
        )
    elif chat_mode == "chain":
        from .chain import get_answer
        return get_answer(query, history)
    else:
        raise ValueError(f"Unknown chat mode {chat_mode}")


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

        history = gr.State([])

        with gr.Tab("ChatGLM"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        elem_id="chat-box",
                        show_label=False
                    ).style(height=800)
                with gr.Column(scale=-1):
                    with gr.Row():
                        max_length = gr.Slider(32, 4096, value=2048, step=1, label="Maximum length", interactive=True)
                        top_p = gr.Slider(0.01, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                        temperature = gr.Slider(0.01, 5, value=0.95, step=0.01, label="Temperature", interactive=True)

                    with gr.Row():
                        query = gr.Textbox(show_label=False, placeholder="Prompts", lines=4).style(container=False)
                        with gr.Row():
                            generate_button = gr.Button("生成")
                            chain_generate_button = gr.Button("知识库问答")
                        chain_generate_button.click(get_answer, inputs=[query, history], outputs=[chatbot])
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

        with gr.Tab("Chain"):
            chain_ui()

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
