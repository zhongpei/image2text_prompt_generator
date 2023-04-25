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
from .textsplitter import ChineseTextSplitter

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

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
def file2doc(file_path: str, max_length=512, min_length=100):
    try:
        if file_path.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(file_path)
            textsplitter = ChineseTextSplitter(pdf=True, max_length=max_length, min_length=min_length)
            doc = loader.load_and_split(textsplitter)
        elif file_path.lower().endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
            textsplitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
            doc = loader.load_and_split(text_splitter=textsplitter)
        else:
            #loader = UnstructuredFileLoader(file_path, mode="elements")
            loader = UnstructuredFileLoader(file_path)
            textsplitter = ChineseTextSplitter(pdf=False, max_length=max_length, min_length=min_length)
            doc = loader.load_and_split(text_splitter=textsplitter)
        return doc
    except Exception as e:
        print(e)
        return


def load_docs(input_path: str | List[str], max_length=512, min_length=100) -> List[Any]:
    docs = []
    print(f"load_docs: {input_path}, {type(input_path)}")
    if isinstance(input_path, list):
        for f in input_path:
            docs += file2doc(f, max_length=max_length, min_length=min_length)

    elif os.path.isfile(input_path):
        docs = [file2doc(input_path, max_length=max_length, min_length=min_length), ]

    elif os.path.isdir(input_path):
        for fn in os.listdir(input_path):
            docs += file2doc(os.path.join(input_path, fn), max_length=max_length, min_length=min_length)

    print(f"load {input_path} ==> docs: {len(docs)}")
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

            )
            self.embeddings.client.save(embedding_dir)
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_dir,

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
            max_length=512,
            min_length=100,
    ):
        docs = load_docs(filepath, max_length=max_length, min_length=min_length)
        if vs_path is None:
            vs_path = os.path.join(VS_ROOT_PATH, vs_id)
        print(f"vector store path: {vs_path} \n docs: {docs}")
        index_file = os.path.join(vs_path, "index.faiss")

        if os.path.isfile(index_file):
            # add doc to exist vector store

            print(f"add doc to exist vector store {index_file}")
            vector_store = FAISS.load_local(vs_path, self.embeddings)
            self.vector_store = vector_store

            if len(docs) > 0:
                vector_store.add_documents(docs)
                vector_store.save_local(vs_path)
                self.vector_store = vector_store
        elif len(docs) > 0:

            if not os.path.exists(vs_path):
                os.makedirs(vs_path, exist_ok=True)
            print(f"create new vector store {vs_path}")

            vector_store = FAISS.from_documents(docs, self.embeddings)
            vector_store.save_local(vs_path)
            self.vector_store = vector_store
            return

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
