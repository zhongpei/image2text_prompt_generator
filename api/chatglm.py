import json
from typing import List

import uvicorn
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from utils.chatglm import models as chatglm_model


def torch_gc():
    import torch

    if torch.cuda.is_available():
        with torch.cuda.device(0):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event('startup')
def init():
    model = chatglm_model.chatglm


class Message(BaseModel):
    role: str
    content: str


class Body(BaseModel):
    messages: List[Message]
    model: str
    stream: bool
    max_tokens: int


chatglm_6b_int4 = {
    "enable": True,
    "model_name": 'chatglm',
    "type": 'chatglm',
    "tokenizer": chatglm_model.chatglm.tokenizer,
    "model": chatglm_model.chatglm.model,

}

models = [chatglm_6b_int4]


@app.post("/chat/completions")
async def completions(body: Body, request: Request):
    global models

    def check_model(model_name):
        for model in models:
            if model_name == model['model_name']:
                return True, model
        return False, None

    exist, model = check_model(body.model)
    if not exist:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Not Implemented")

    if not model["enable"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Disabled")

    question = body.messages[-1]
    if question.role == 'user':
        question = question.content
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")

    user_question = ''

    history = []
    for message in body.messages:
        if message.role == 'user':
            user_question = message.content
        elif message.role == 'system' or message.role == 'assistant':
            assistant_answer = message.content
            history.append((user_question, assistant_answer))

    completion_text = """The following is a conversation with an AI assistant.
The assistant is helpful, creative, clever, and very friendly. The assistant is familiar with various languages in the world.
Human: Hello, who are you?
AI: I am an AI assistant. How can I help you today?
Human: 没什么
AI: 好的, 如果有什么需要, 随时告诉我"""
    for message in body.messages:
        if message.role == 'user':
            completion_text += "\nHuman: " + message.content
        elif message.role == 'assistant':
            completion_text += "\nAI: " + message.content
    completion_text += "\nAI: "

    async def eval_chatglm():
        if body.stream:
            for response in model["model"].stream_chat(
                    model["tokenizer"],
                    question,
                    history,
                    max_length=max(2048, body.max_tokens)
            ):
                if await request.is_disconnected():
                    torch_gc()
                    return
                yield json.dumps({"response": response[0]})
            yield "[DONE]"
        else:
            response, _ = model["model"].chat(
                model["tokenizer"],
                question,
                history,
                max_length=max(2048, body.max_tokens)
            )
            yield json.dumps({"response": response[0]})
        torch_gc()

    return EventSourceResponse(eval_chatglm())


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, app_dir=".")
