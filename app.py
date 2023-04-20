import gradio as gr
import torch
from config import settings
from utils.chatglm import chat2text
from utils.exif import get_image_info
from utils.generator import generate_prompt
from utils.image2text import git_image2text, w14_image2text, clip_image2text
from utils.translate import en2zh as translate_en2zh
from utils.translate import zh2en as translate_zh2en
from ui.chat import chatglm_ui
import click
from imagetools.ui import image_tools_ui
from ui.image2text import image2text_settings_ui

device = "cuda" if torch.cuda.is_available() else "cpu"


def text_generate_prompter(
        plain_text,
        model_name='microsoft',
        prompt_min_length=60,
        prompt_max_length=75,
        prompt_num_return_sequences=8,
):
    result = generate_prompt(
        plain_text=plain_text,
        model_name=model_name,
        min_length=prompt_min_length,
        max_length=prompt_max_length,
        num_return_sequences=prompt_num_return_sequences
    )
    empty_cache()
    return "\n\n".join(result), "\n\n".join(translate_en2zh(line) for line in result if len(line) > 0)


def image_generate_prompter(
        bclip_text,
        w14_text,
        model_name='microsoft',
        prompt_min_length=60,
        prompt_max_length=75,
        prompt_num_return_sequences=8,

):
    result = generate_prompt(
        plain_text=bclip_text,
        model_name=model_name,
        min_length=prompt_min_length,
        max_length=prompt_max_length,
        num_return_sequences=prompt_num_return_sequences
    )
    prompter_list = ["{},{}".format(line.strip(), w14_text.strip()) for line in result if len(line) > 0]
    prompter_zh_list = [
        "{},{}".format(translate_en2zh(line.strip()), translate_en2zh(w14_text.strip())) for line in
        result if len(line) > 0
    ]
    empty_cache()
    return "\n\n".join(prompter_list), "\n\n".join(prompter_zh_list)


def translate_input(text: str, chatglm_text: str) -> str:
    if chatglm_text is not None and len(chatglm_text) > 0:
        return translate_zh2en(chatglm_text)
    return translate_zh2en(text)


def empty_cache(force_clear_cache: bool = False):
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if force_clear_cache:
            try:
                from numba import cuda
                current_cuda = cuda.get_current_device()
                current_cuda.reset()
            except:
                pass


def ui(enable_chat: bool = False):
    with gr.Blocks(title="Prompt生成器") as block:
        with gr.Column():
            with gr.Row():
                force_clear_cache = gr.Checkbox(False, label='强制清显存', width=60)
                empty_cache_btn = gr.Button('清显存')

            with gr.Tab('文本生成'):
                with gr.Row():
                    input_text = gr.Textbox(lines=6, label='你的想法', placeholder='在此输入内容...')
                    chatglm_output = gr.Textbox(lines=6, label='ChatGLM', placeholder='在此输入内容...')

                    translate_output = gr.Textbox(lines=6, label='翻译结果(Prompt输入)')

                output = gr.Textbox(lines=6, label='优化的 Prompt')
                output_zh = gr.Textbox(lines=6, label='优化的 Prompt(zh)')
                with gr.Row():
                    chatglm_btn = gr.Button('召唤ChatGLM')
                    translate_btn = gr.Button('翻译')
                    generate_prompter_btn = gr.Button('优化Prompt')

            with gr.Tab('从图片中生成'):
                with gr.Row():
                    input_image = gr.Image(type='pil')
                    exif_info = gr.HTML()
                output_blip_or_clip = gr.Textbox(label='生成的 Prompt', lines=4)
                output_w14 = gr.Textbox(label='W14的 Prompt', lines=4)

                with gr.Accordion('W14', open=False):
                    w14_raw_output = gr.Textbox(label="Output (raw string)")
                    w14_booru_output = gr.Textbox(label="Output (booru string)")
                    w14_rating_output = gr.Label(label="Rating")
                    w14_characters_output = gr.Label(label="Output (characters)")
                    w14_tags_output = gr.Label(label="Output (tags)")
                output_img_prompter = gr.Textbox(lines=6, label='优化的 Prompt')
                output_img_prompter_zh = gr.Textbox(lines=6, label='优化的 Prompt(zh)')
                with gr.Row():
                    img_exif_btn = gr.Button('EXIF')
                    img_blip_btn = gr.Button('GIT图片转描述')
                    img_w14_btn = gr.Button('W14图片转描述')
                    img_clip_btn = gr.Button('CLIP图片转描述')
                    img_prompter_btn = gr.Button('优化Prompt')

            with gr.Tab('参数设置'):
                with gr.Accordion('Prompt优化参数', open=True):
                    prompt_mode_name = gr.Radio(
                        [
                            'microsoft',
                            'mj',
                            'gpt2_650k',
                            'gpt_neo_125m',
                        ],
                        value=settings.generator.default_model_name,
                        label='model_name'
                    )
                    prompt_min_length = gr.Slider(1, 512, 100, label='min_length', step=1)
                    prompt_max_length = gr.Slider(1, 512, 200, label='max_length', step=1)
                    prompt_num_return_sequences = gr.Slider(
                        1,
                        30,
                        value=settings.generator.default_num_return_sequences,
                        label='num_return_sequences',
                        step=1
                    )
                git_max_length, clip_mode_type, clip_model_name, wd14_model_name, wd14_general_threshold, wd14_character_threshold = image2text_settings_ui()

            if enable_chat:
                chatglm_ui()
            if settings.image_tools.enable:
                image_tools_ui(
                    clip_mode_type=clip_mode_type,
                    clip_model_name=clip_model_name,
                    wd14_model=wd14_model_name,
                    wd14_general_threshold=wd14_general_threshold,
                    wd14_character_threshold=wd14_character_threshold,
                )

        empty_cache_btn.click(fn=empty_cache, inputs=force_clear_cache)
        img_prompter_btn.click(
            fn=image_generate_prompter,
            inputs=[
                output_blip_or_clip,
                output_w14,
                prompt_mode_name,
                prompt_min_length,
                prompt_max_length,
                prompt_num_return_sequences,

            ],
            outputs=[output_img_prompter, output_img_prompter_zh]
        )
        chatglm_btn.click(
            fn=chat2text,
            inputs=input_text,
            outputs=chatglm_output,
        )
        translate_btn.click(
            fn=translate_input,
            inputs=[input_text, chatglm_output],
            outputs=translate_output
        )

        generate_prompter_btn.click(
            fn=text_generate_prompter,
            inputs=[
                translate_output,
                prompt_mode_name,
                prompt_min_length,
                prompt_max_length,
                prompt_num_return_sequences,
            ],
            outputs=[output, output_zh]
        )
        img_w14_btn.click(
            fn=w14_image2text,
            inputs=[input_image, wd14_model_name, wd14_general_threshold, wd14_character_threshold],
            outputs=[
                output_w14,
                w14_raw_output,
                w14_booru_output,
                w14_rating_output,
                w14_characters_output,
                w14_tags_output
            ]
        )

        img_blip_btn.click(
            fn=git_image2text,
            inputs=[input_image, git_max_length],
            outputs=output_blip_or_clip
        )
        img_clip_btn.click(
            fn=clip_image2text,
            inputs=[input_image, clip_mode_type, clip_model_name],
            outputs=output_blip_or_clip
        )

        img_exif_btn.click(
            fn=get_image_info,
            inputs=input_image,
            outputs=exif_info
        )
    return block


@click.command()
@click.option('--chat', is_flag=True, help='Enable chat.', default=False)
@click.option('--queue', is_flag=True, help='Enable queue.', default=False)
def main(chat, queue):
    block = ui(enable_chat=chat or settings.chatglm.enable_chat)
    block.queue(max_size=settings.server.queue_size).launch(
        show_api=settings.server.show_api,
        enable_queue=queue or settings.server.enable_queue,
        debug=settings.server.debug,
        share=False,
        server_name=settings.server.host,
        server_port=settings.server.port,
    )


if __name__ == '__main__':
    main()
