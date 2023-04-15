import gradio as gr
import torch

from utils.exif import get_image_info
from utils.generator import generate_prompt
from utils.image2text import git_image2text, w14_image2text, clip_image2text
from utils.translate import en2zh as translate_en2zh
from utils.translate import zh2en as translate_zh2en

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
    return result, "\n".join(translate_en2zh(line) for line in result.split("\n") if len(line) > 0)


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
    prompter_list = ["{},{}".format(line.strip(), w14_text.strip()) for line in result.split("\n") if len(line) > 0]
    prompter_zh_list = [
        "{},{}".format(translate_en2zh(line.strip()), translate_en2zh(w14_text.strip())) for line in
        result.split("\n") if len(line) > 0
    ]
    return "\n".join(prompter_list), "\n".join(prompter_zh_list)


with gr.Blocks(title="Prompt生成器") as block:
    with gr.Column():
        with gr.Tab('文本生成'):
            with gr.Row():
                input_text = gr.Textbox(lines=6, label='你的想法', placeholder='在此输入内容...')
                translate_output = gr.Textbox(lines=6, label='翻译结果(Prompt输入)')

            output = gr.Textbox(lines=6, label='优化的 Prompt')
            output_zh = gr.Textbox(lines=6, label='优化的 Prompt(zh)')
            with gr.Row():
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
                img_blip_btn = gr.Button('BLIP图片转描述')
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
                    ],
                    value='gpt2_650k',
                    label='model_name'
                )
                prompt_min_length = gr.Slider(1, 512, 100, label='min_length', step=1)
                prompt_max_length = gr.Slider(1, 512, 200, label='max_length', step=1)
                prompt_num_return_sequences = gr.Slider(1, 30, 6, label='num_return_sequences', step=1)

            with gr.Accordion('BLIP参数', open=True):
                blip_max_length = gr.Slider(1, 512, 100, label='max_length', step=1)
            with gr.Accordion('CLIP参数', open=True):
                clip_mode_type = gr.Radio(['best', 'classic', 'fast', 'negative'], value='best', label='mode_type')
                clip_model_name = gr.Radio(['vit_h_14', 'vit_l_14', ], value='vit_h_14', label='model_name')
            with gr.Accordion('WD14参数', open=True):
                image2text_model = gr.Radio(
                    [
                        "SwinV2",
                        "ConvNext",
                        "ConvNextV2",
                        "ViT",
                    ],
                    value="ConvNextV2",
                    label="Model"
                )
                general_threshold = gr.Slider(
                    0,
                    1,
                    step=0.05,
                    value=0.35,
                    label="General Tags Threshold",
                )
                character_threshold = gr.Slider(
                    0,
                    1,
                    step=0.05,
                    value=0.85,
                    label="Character Tags Threshold",
                )
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
    translate_btn.click(
        fn=translate_zh2en,
        inputs=input_text,
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
        inputs=[input_image, image2text_model, general_threshold, character_threshold],
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
        inputs=[input_image, blip_max_length],
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
block.queue(max_size=64).launch(show_api=False, enable_queue=False, debug=True, share=False, server_name='127.0.0.1')
