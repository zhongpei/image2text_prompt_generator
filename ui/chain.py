import os
import shutil

import gradio as gr
import nltk
import torch
from datetime import datetime
from utils.chain import LocalDocQA

device = "cuda" if torch.cuda.is_available() else "cpu"

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6

# LLM input history length
LLM_HISTORY_LEN = 3

VS_ROOT_PATH = "./data/chain/vector_store/"

UPLOAD_ROOT_PATH = "./data/chain/upload/"

local_doc_qa = LocalDocQA()


def get_vs_list(vs_path):
    vlist = []
    if not os.path.exists(vs_path):
        return vlist
    vlist = os.listdir(vs_path)
    vlist = [v for v in vlist if v != ".keep"]
    return vlist


def get_answer(query, history):
    resp, history = local_doc_qa.get_knowledge_based_answer(
        query=query,
        chat_history=history
    )
    source = "".join(
        [f"""<details> <summary>出处 {i + 1}</summary>
{doc.page_content}
<b>所属文件：</b>{doc.metadata["source"]}
</details>""" for i, doc in enumerate(resp["source_documents"])
         ]
    )
    history[-1][-1] += source

    return history


def init_model():
    try:
        local_doc_qa.init_cfg()

        return """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
    except Exception as e:
        print(e)
        return """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""


def reinit_model(embedding_model, llm_history_len, top_k, history):
    try:
        local_doc_qa.init_cfg(
            embedding_model=embedding_model,
            llm_history_len=llm_history_len,
        )
        model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
    except Exception as e:
        print(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
    return history + [[None, model_status]]


def upload_files(files):
    filelist = []
    for file in files:
        filename = os.path.split(file.name)[-1]
        ext = os.path.splitext(filename)[-1]
        filename = filename.replace(" ", "_")
        filename = "{}_{}.{}".format(filename, datetime.now().strftime("%Y%m%d%H%M%S"), ext)
        shutil.move(file.name, os.path.join(UPLOAD_ROOT_PATH, filename))
        filelist.append(os.path.join(UPLOAD_ROOT_PATH, filename))
    return filelist


def init_vector_store(vs_id: str, input_files):
    vs_path = os.path.join(VS_ROOT_PATH, vs_id)

    filelist = upload_files(input_files)

    if local_doc_qa.is_initialized():
        local_doc_qa.init_knowledge_vector_store(
            vs_id=vs_id,
            filepath=filelist,
            vs_path=vs_path
        )

    return None


def change_mode(mode):
    if mode == "知识库问答":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def add_vs_name(vs_name):
    vs_list = get_vs_list(VS_ROOT_PATH)

    if vs_name in vs_list:
        return f"知识库{vs_name}已存在，请重新输入", vs_list
    vs_list.append(vs_name)
    return f"知识库{vs_name}创建成功", list(set(vs_list)), gr.update(visible=True, value=vs_list)


def chain_upload_ui(select_vs, result):
    with gr.Column(visible=True):
        # load_vs = gr.Button("加载知识库")
        gr.Markdown("向知识库中添加文件")
        with gr.Tab("上传文件"):
            files = gr.File(
                label="添加文件",
                file_types=['.txt', '.md', '.docx', '.pdf'],
                file_count="multiple",
                show_label=False
            )
            load_file_button = gr.Button("上传文件并加载知识库")
        with gr.Tab("上传文件夹"):
            folder_files = gr.File(
                label="添加文件",
                file_count="directory",
                show_label=False
            )
            load_folder_button = gr.Button("上传文件夹并加载知识库")

    # 将上传的文件保存到content文件夹下,并更新下拉框
    load_file_button.click(
        init_vector_store,
        show_progress=True,
        inputs=[select_vs, files],
        outputs=[result],
    )
    load_folder_button.click(
        init_vector_store,
        show_progress=True,
        inputs=[select_vs, folder_files],
        outputs=[result],
    )


def chain_ui(chatbot, query):
    init_model()
    vs_list = gr.State(get_vs_list(VS_ROOT_PATH))

    result = gr.Textbox(label="结果", lines=1, interactive=False)
    with gr.Accordion("加载知识库", open=False):
        select_vs = gr.Dropdown(
            vs_list.value,
            label="请选择要加载的知识库",
            interactive=True,
            value=vs_list.value[0] if len(vs_list.value) > 0 else None
        )

        chain_upload_ui(select_vs=select_vs, result=result)

    with gr.Accordion("新建知识库", open=False):
        vs_name = gr.Textbox(
            label="请输入新建知识库名称",
            lines=1,
            interactive=True
        )
        vs_add_button = gr.Button(value="添加知识库")
        vs_add_button.click(
            fn=add_vs_name,
            inputs=vs_name,
            outputs=[result, vs_list, select_vs]
        )
