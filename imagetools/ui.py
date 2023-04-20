from autocrop.cli import main as autocrop_main
from .remove_backgroud import remove_background
import gradio as gr
import os
from .mesh_face import mesh_face, mesh_hand
from config import settings
from .tags import load_translated_tags
from .tags import gen_clip_tags_files


def bz_autocrop(input_dir, output_dir, reject_dir, height=512, width=512, facePercent=50):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(reject_dir):
        os.mkdir(reject_dir)

    result = autocrop_main(
        input_d=input_dir,
        output_d=output_dir,
        reject_d=reject_dir,
        fheight=height,
        fwidth=width,
        facePercent=facePercent
    )
    return result


def bz_mesh(input_dir, output_dir, max_faces=1, thickness=10, circle_radius=10, mesh_type='face'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if mesh_type == "face":
        result = mesh_face(
            image_path=input_dir,
            output_path=output_dir,
            max_num_faces=max_faces,
            thickness=thickness,
            circle_radius=circle_radius

        )
    if mesh_type == "hand":
        result = mesh_hand(
            image_path=input_dir,
            output_path=output_dir,
            thickness=thickness,
            circle_radius=circle_radius
        )
    else:
        raise Exception("mesh_type must be face or hand")
    return result


def remove_background_func(input_dir, output_dir, background_type, background_mode):
    remove_background(
        image_path=input_dir,
        output_path=output_dir,
        background_type=background_type,
        ckpt=settings.image_tools.get('transparent_background_model_path'),
        fast=True if background_mode == "fast" else False,
        jit=False,
    )


def rename_func(input_dir, postfix, source, target):
    import re
    files = os.listdir(input_dir)
    for file in files:
        if file.endswith(postfix):
            new_file = re.sub(source, target, file)
            os.rename(os.path.join(input_dir, file), os.path.join(input_dir, new_file))
    return "rename success"


def load_translated_tags_fn(input_dir: str):
    tags, zh_tags = load_translated_tags(input_dir)
    return dict(tags), dict(zh_tags)


def image_tools_ui(clip_mode_type, clip_model_name):
    with gr.Tab("image tools"):
        with gr.Tab("remove background(扣背)"):
            remove_input_dir = gr.Textbox(label='input_dir')
            remove_output_dir = gr.Textbox(label='output_dir')
            background_type = gr.Radio(choices=["white", "green", "rgba"], label="background_type", value="white")
            background_mode = gr.Radio(choices=["fast", "base", ], label="background_mode", value="base")
            remove_background_button = gr.Button("remove")

        with gr.Tab("mesh face(糊脸)"):
            mash_input_dir = gr.Textbox(label='input_dir')
            mash_output_dir = gr.Textbox(label='output_dir')
            max_faces = gr.Slider(0, 20, value=1, label='max_faces', step=1)
            thickness = gr.Slider(0, 20, value=10, label='thickness', step=1)
            circle_radius = gr.Slider(0, 100, value=15, label='circle_radius', step=1)
            mesh_type = gr.Radio(choices=["face", ], label="mesh_type", value="face")

            mesh_face_button = gr.Button("mesh")

        with gr.Tab("autocrop(大头)"):
            input_dir = gr.Textbox(label='input_dir')
            output_dir = gr.Textbox(label='output_dir')
            reject_dir = gr.Textbox(label='reject_dir')
            height = gr.Slider(0, 1024, value=512, label='height', step=1)
            width = gr.Slider(0, 1024, value=512, label='width', step=1)
            facePercent = gr.Slider(0, 100, value=40, label='facePercent', step=1)
            autocrop_button = gr.Button("autocrop")

        with gr.Tab("rename(改名)"):
            rename_input_dir = gr.Textbox(label='input_dir')
            rename_postfix = gr.Textbox(label='文件后缀', value="txt")
            rename_replace_source = gr.Textbox(label='replace_source', value=r"\d+-\d+-")
            rename_replace_target = gr.Textbox(label='replace_targete', value="")
            rename_btn = gr.Button("rename")
        with gr.Tab("tags(标签)"):
            with gr.Row():
                tags_input_dir = gr.Textbox(label='input_dir')
                translate_tags_btn = gr.Button("load tags(加载标签)")

            with gr.Row():
                gen_tags_mode = gr.Radio(["top", "down", "center", "convert", "random"], label="tags mode", value="top")
                gen_clip_tags_btn = gr.Button("clip tags(生成CLIP标签签)")


            with gr.Accordion("tags", open=False):
                tags_label = gr.Label("tags")
                tags_zh_label = gr.Label("tags_zh")

        text_output = gr.Textbox(label="result", lines=1, max_lines=100)

        gen_clip_tags_btn.click(
            gen_clip_tags_files,
            inputs=[tags_input_dir, gen_tags_mode, clip_mode_type, clip_model_name],
            outputs=text_output
        )

        translate_tags_btn.click(
            load_translated_tags_fn,
            inputs=tags_input_dir,
            outputs=[tags_label, tags_zh_label]
        )

        rename_btn.click(
            rename_func,
            inputs=[rename_input_dir, rename_postfix, rename_replace_source, rename_replace_target],
            outputs=text_output,
        )
        mesh_face_button.click(
            bz_mesh,
            inputs=[mash_input_dir, mash_output_dir, max_faces, thickness, circle_radius, mesh_type],
            outputs=text_output
        )

        remove_background_button.click(
            remove_background_func,
            inputs=[remove_input_dir, remove_output_dir, background_type, background_mode],
            outputs=text_output

        )
        autocrop_button.click(
            bz_autocrop,
            inputs=[input_dir, output_dir, reject_dir, height, width, facePercent],
            outputs=text_output
        )
