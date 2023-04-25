import os
import re
from typing import Any
from typing import List, Union

import sentence_transformers
import torch
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import BaseLLM
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


def file2doc(file_path: str):
    try:
        if file_path.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(file_path)
            textsplitter = ChineseTextSplitter(pdf=True)
            doc = loader.load_and_split(textsplitter)
        else:
            loader = UnstructuredFileLoader(file_path, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False)
            doc = loader.load_and_split(text_splitter=textsplitter)
        return doc
    except Exception as e:
        print(e)
        return


def load_docs(path: str | List[str]) -> List[Any]:
    docs = []
    print(f"load_docs: {path}, {type(path)}")
    if isinstance(path, list):
        docs = [file2doc(f) for f in path]

    if os.path.isfile(path):
        docs = [file2doc(path), ]

    if os.path.isdir(path):
        docs = [file2doc(os.path.join(path, f)) for f in os.listdir(path)]
    print(f"load {path} ==> docs: {docs}")
    return docs


class LocalDocQA:
    llm: BaseLLM = None
    embeddings: HuggingFaceEmbeddings = None
    vector_store: FAISS = None

    def init_cfg(
            self,
            embedding_model: str = EMBEDDING_MODEL,
            embedding_device: str = EMBEDDING_DEVICE,
            llm_history_len: int = LLM_HISTORY_LEN,

    ):
        self.llm = chatglm_models.chatglm

        self.llm.history_len = llm_history_len

        embedding_dir = os.path.join("./models", embedding_model)
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_dict[embedding_model],
                device=embedding_device,
            )
            self.embeddings.client.save(embedding_dir)
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_dir,
                device=embedding_device,
            )

        # self.embeddings.client = sentence_transformers.SentenceTransformer(
        #     self.embeddings.model_name,
        #     device=embedding_device
        # )

    def is_active(self):
        return self.vector_store is not None

    def is_initialized(self):
        return self.llm is not None and self.embeddings is not None

    def init_knowledge_vector_store(
            self,
            vs_id: str,
            filepath: Union[str, List[str]],
            vs_path: str = None,
    ):
        docs = load_docs(filepath)
        if vs_path is None:
            vs_path = os.path.join(VS_ROOT_PATH, vs_id)
        print(f"vector store path: {vs_path} \n docs: {docs}")

        if os.path.isfile(os.path.join(vs_path, "index.faiss")):
            # add doc to exist vector store
            index_file = os.path.join(vs_path, "index.faiss")
            print(f"add doc to exist vector store {index_file}")
            vector_store = FAISS.load_local(vs_path, self.embeddings)
            vector_store.add_documents(docs)
        else:
            if not os.path.exists(vs_path):
                os.makedirs(vs_path, exist_ok=True)
            print(f"create new vector store {vs_path}")

            vector_store = FAISS.from_documents(docs, self.embeddings)

        vector_store.save_local(vs_path)
        self.vector_store = vector_store

    def get_knowledge_based_answer(
            self,
            query,
            chat_history=None,
            top_k: int = VECTOR_SEARCH_TOP_K,
    ):
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
        if chat_history is None:
            chat_history = []

        self.llm.history = chat_history

        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": top_k}),
            prompt=prompt
        )
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        knowledge_chain.return_source_documents = True

        result = knowledge_chain({"query": query})
        self.llm.history[-1][0] = query
        return result, self.llm.history
