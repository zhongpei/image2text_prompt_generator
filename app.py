import gradio as gr
import torch
import mdtex2html
from utils.exif import get_image_info
from utils.generator import generate_prompt
from utils.image2text import git_image2text, w14_image2text, clip_image2text
from utils.translate import en2zh as translate_en2zh
from utils.translate import zh2en as translate_zh2en
from utils.chatglm import chat2text
from utils.chatglm import models as chatglm_models

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


def translate_input(text: str, chatglm_text: str) -> str:
    if chatglm_text is not None and len(chatglm_text) > 0:
        return translate_zh2en(chatglm_text)
    return translate_zh2en(text)


with gr.Blocks(title="Prompt生成器") as block:
    with gr.Column():
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
                        max_length = gr.Slider(32, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
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
                        'gpt_neo_125m',
                    ],
                    value='gpt2_650k',
                    label='model_name'
                )
                prompt_min_length = gr.Slider(1, 512, 100, label='min_length', step=1)
                prompt_max_length = gr.Slider(1, 512, 200, label='max_length', step=1)
                prompt_num_return_sequences = gr.Slider(1, 30, 8, label='num_return_sequences', step=1)

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
    chatglm_btn.click(
        fn=chatglm_models.chatglm.generator_image_text,
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
block.queue(max_size=64).launch(show_api=False, enable_queue=True, debug=True, share=False, server_name='127.0.0.1')
