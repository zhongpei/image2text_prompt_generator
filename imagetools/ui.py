from autocrop.cli import main as autocrop_main
from .remove_backgroud import remove_background
import gradio as gr
import os
from .mesh_face import mesh_face, mesh_hand
from config import settings


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


def remove_background_func(input_dir, output_dir, background_type):
    if os.path.exists(settings.image_tools.transparent_background_path):
        bin_path = settings.image_tools.transparent_background_path
    elif os.path.exists(os.path.join("venv", "Scripts", "transparent-background.exe")):
        bin_path = os.path.join("venv", "Scripts", "transparent-background.exe")
    elif os.path.exists(os.path.join("venv", "bin", "transparent-background")):
        bin_path = os.path.join("venv", "bin", "transparent-background")
    else:
        bin_path = "transparent-background"
    remove_background(input_dir, output_dir, background_type, bin_path=bin_path)


def image_tools_ui():
    with gr.Tab("image tools"):
        with gr.Tab("remove background(扣背)"):
            remove_input_dir = gr.Textbox(label='input_dir')
            remove_output_dir = gr.Textbox(label='output_dir')
            background_type = gr.Radio(choices=["white", "green"], label="background_type", value="white")
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
            facePercent = gr.Slider(0, 100, value=50, label='facePercent', step=1)
            autocrop_button = gr.Button("autocrop")
        text_output = gr.Textbox(label="result")
        mesh_face_button.click(
            bz_mesh,
            inputs=[mash_input_dir, mash_output_dir, max_faces, thickness, circle_radius, mesh_type],
            outputs=text_output
        )

        remove_background_button.click(
            remove_background_func,
            inputs=[remove_input_dir, remove_output_dir, background_type],
            outputs=text_output

        )
        autocrop_button.click(
            bz_autocrop,
            inputs=[input_dir, output_dir, reject_dir, height, width, facePercent],
            outputs=text_output
        )
