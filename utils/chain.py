import datetime
import os
import re
from typing import List

import sentence_transformers
import torch
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from config import settings
from utils.chatglm import models as chatglm_models

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False
# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6

# LLM input history length
LLM_HISTORY_LEN = 3

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

VS_ROOT_PATH = "./data/chain/vector_store/"

UPLOAD_ROOT_PATH = "./data/chain/upload/"

LLM_MODEL = settings.chatglm.model if settings.chatglm.model else "THUDM/chatglm-6b-int4"


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


def load_file(filepath):
    if filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs


class LocalDocQA:
    llm: object = None
    embeddings: HuggingFaceEmbeddings = None
    top_k: int = VECTOR_SEARCH_TOP_K

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device: str = EMBEDDING_DEVICE,
                 llm_history_len: int = LLM_HISTORY_LEN,
                 top_k: int = VECTOR_SEARCH_TOP_K,
                 ):
        self.llm = chatglm_models.chatglm

        self.llm.history_len = llm_history_len

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model], )
        self.embeddings.client = sentence_transformers.SentenceTransformer(
            self.embeddings.model_name,
            device=embedding_device
        )
        self.top_k = top_k

    def init_knowledge_vector_store(
            self,
            filepath: str or List[str],
            vs_path: str or os.PathLike = None
    ):
        loaded_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath)
                    print(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += load_file(fullfilepath)
                        print(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} 未能成功加载")
        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    print(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")

        if vs_path and os.path.isdir(vs_path):
            vector_store = FAISS.load_local(vs_path, self.embeddings)
            vector_store.add_documents(docs)
        else:
            if not vs_path:
                vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
            vector_store = FAISS.from_documents(docs, self.embeddings)

        vector_store.save_local(vs_path)
        return vs_path if len(docs) > 0 else None, loaded_files

    def get_knowledge_based_answer(self,
                                   query,
                                   vs_path,
                                   chat_history=[], ):
        prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

    已知内容:
    {context}

    问题:
    {question}"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        self.llm.history = chat_history
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": self.top_k}),
            prompt=prompt
        )
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        knowledge_chain.return_source_documents = True

        result = knowledge_chain({"query": query})
        self.llm.history[-1][0] = query
        return result, self.llm.history
