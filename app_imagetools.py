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
        image_tools_ui()

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
