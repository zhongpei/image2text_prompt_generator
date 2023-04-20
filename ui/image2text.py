import gradio as gr
from config import settings


def image2text_settings_ui():
    with gr.Accordion('GIT参数', open=True):
        git_max_length = gr.Slider(1, 512, 200, label='max_length', step=1)
    with gr.Accordion('CLIP参数', open=True):
        clip_mode_type = gr.Radio(
            ['best', 'classic', 'fast', 'negative'],
            value=settings.clip.default_model_type,
            label='mode_type'
        )
        clip_model_name = gr.Radio(
            ['vit_h_14', 'vit_l_14', ],
            value=settings.clip.default_model_name,
            label='model_name'
        )
    with gr.Accordion('WD14参数', open=True):
        wd14_model_name = gr.Radio(
            [
                "SwinV2",
                "ConvNext",
                "ConvNextV2",
                "ViT",
            ],
            value=settings.wd14.default_model_name,
            label="Model"
        )
        wd14_general_threshold = gr.Slider(
            0,
            1,
            step=0.05,
            value=settings.wd14.default_general_threshold,
            label="General Tags Threshold",
        )
        wd14_character_threshold = gr.Slider(
            0,
            1,
            step=0.05,
            value=settings.wd14.default_character_threshold,
            label="Character Tags Threshold",
        )
    return git_max_length, clip_mode_type, clip_model_name, wd14_model_name, wd14_general_threshold, wd14_character_threshold
