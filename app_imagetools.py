import gradio as gr
import torch
import click
from config import settings
from imagetools.ui import image_tools_ui
import signal

device = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option("--port", type=int, default=None, help="server port")
def ui(port):
    with gr.Blocks(title="图片处理工具") as block:
        image_tools_ui(
            clip_mode_type=settings.clip.default_model_type,
            clip_model_name=settings.clip.default_model_name,
            wd14_model=settings.wd14.default_model_name,
            wd14_general_threshold=settings.wd14.default_general_threshold,
            wd14_character_threshold=settings.wd14.default_character_threshold,
        )

    block.queue(max_size=settings.server.queue_size).launch(
        show_api=settings.server.show_api,
        enable_queue=settings.server.enable_queue,
        debug=settings.server.debug,
        share=False,
        server_name=settings.server.host,
        server_port=port or settings.server.port,
    )


def signal_handler(sig, frame):
    gr.close_all()
    exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    ui()
