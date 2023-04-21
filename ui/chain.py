import nltk
import torch

from utils.chain import LocalDocQA

import gradio as gr
import os
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6

# LLM input history length
LLM_HISTORY_LEN = 3

VS_ROOT_PATH = "./data/chain/vector_store/"

UPLOAD_ROOT_PATH = "./data/chain/upload/"

local_doc_qa = LocalDocQA()


def get_vs_list():
    vlist = []
    if not os.path.exists(VS_ROOT_PATH):
        return vlist
    vlist = ["新建知识库"] + os.listdir(VS_ROOT_PATH)
    vlist = [v for v in vlist if v != ".keep"]
    return vlist


def get_answer(query, vs_path, history, mode):
    if vs_path and mode == "知识库问答":
        resp, history = local_doc_qa.get_knowledge_based_answer(
            query=query, vs_path=vs_path, chat_history=history)
        source = "".join(
            [f"""<details> <summary>出处 {i + 1}</summary>
{doc.page_content}
<b>所属文件：</b>{doc.metadata["source"]}
</details>""" for i, doc in enumerate(resp["source_documents"])
             ]
        )
        history[-1][-1] += source
    else:
        resp = local_doc_qa.llm._call(query)
        history = history + [[query, resp + (
            "\n\n当前知识库为空，如需基于知识库进行问答，请先加载知识库后，再进行提问。" if mode == "知识库问答" else "")]]
    return history, ""


def init_model():
    try:
        local_doc_qa.init_cfg()
        local_doc_qa.llm._call("你好")
        return """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
    except Exception as e:
        print(e)
        return """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""


def reinit_model(embedding_model, llm_history_len, top_k, history):
    try:
        local_doc_qa.init_cfg(
            embedding_model=embedding_model,
            llm_history_len=llm_history_len,

            top_k=top_k
        )
        model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
    except Exception as e:
        print(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
    return history + [[None, model_status]]


def get_vector_store(vs_id, files, history):
    vs_path = VS_ROOT_PATH + vs_id
    filelist = []
    for file in files:
        filename = os.path.split(file.name)[-1]
        shutil.move(file.name, UPLOAD_ROOT_PATH + filename)
        filelist.append(UPLOAD_ROOT_PATH + filename)
    if local_doc_qa.llm and local_doc_qa.embeddings:
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path)
        if len(loaded_files):
            file_status = f"已上传 {'、'.join([os.path.split(i)[-1] for i in loaded_files])} 至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    return vs_path, None, history + [[None, file_status]]


def change_vs_name_input(vs_id):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), VS_ROOT_PATH + vs_id


def change_mode(mode):
    if mode == "知识库问答":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def add_vs_name(vs_name, vs_list, chatbot):
    if vs_name in vs_list:
        chatbot = chatbot + [[None, "与已有知识库名称冲突，请重新选择其他名称后提交"]]
        return gr.update(visible=True), vs_list, chatbot
    else:
        chatbot = chatbot + [
            [None, f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """]
        ]
        return gr.update(visible=True, choices=vs_list + [vs_name], value=vs_name), vs_list + [vs_name], chatbot


def chain_ui(mode, chatbot, query):
    init_model()
    vs_list = gr.State(get_vs_list())
    vs_path, file_status = gr.State(""), gr.State(""),
    vs_setting = gr.Accordion("配置知识库", open=False)
    mode.change(
        fn=change_mode,
        inputs=mode,
        outputs=vs_setting
    )
    with vs_setting:
        select_vs = gr.Dropdown(
            vs_list.value,
            label="请选择要加载的知识库",
            interactive=True,
            value=vs_list.value[0] if len(vs_list.value) > 0 else None
        )
        vs_name = gr.Textbox(
            label="请输入新建知识库名称",
            lines=1,
            interactive=True)
        vs_add = gr.Button(value="添加至知识库选项")
        vs_add.click(
            fn=add_vs_name,
            inputs=[vs_name, vs_list, chatbot],
            outputs=[select_vs, vs_list, chatbot]
        )

        file2vs = gr.Column(visible=False)
        with file2vs:
            # load_vs = gr.Button("加载知识库")
            gr.Markdown("向知识库中添加文件")
            with gr.Tab("上传文件"):
                files = gr.File(label="添加文件",
                                file_types=['.txt', '.md', '.docx', '.pdf'],
                                file_count="multiple",
                                show_label=False
                                )
                load_file_button = gr.Button("上传文件并加载知识库")
            with gr.Tab("上传文件夹"):
                folder_files = gr.File(label="添加文件",
                                       # file_types=['.txt', '.md', '.docx', '.pdf'],
                                       file_count="directory",
                                       show_label=False
                                       )
                load_folder_button = gr.Button("上传文件夹并加载知识库")
        # load_vs.click(fn=)
        select_vs.change(
            fn=change_vs_name_input,
            inputs=select_vs,
            outputs=[vs_name, vs_add, file2vs, vs_path]
        )
        # 将上传的文件保存到content文件夹下,并更新下拉框
        load_file_button.click(get_vector_store,
                               show_progress=True,
                               inputs=[select_vs, files, chatbot],
                               outputs=[vs_path, files, chatbot],
                               )
        load_folder_button.click(get_vector_store,
                                 show_progress=True,
                                 inputs=[select_vs, folder_files, chatbot],
                                 outputs=[vs_path, folder_files, chatbot],
                                 )
        query.submit(
            get_answer,
            [query, vs_path, chatbot, mode],
            [chatbot, query],
        )
