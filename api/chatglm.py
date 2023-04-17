import json
from typing import List

import uvicorn
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from utils.chatglm import models as chatglm_model
from utils.generator import generate_prompt
from utils.translate import zh2en
from utils.chatglm import chat2text


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


class Message(BaseModel):
    role: str
    content: str


class Body(BaseModel):
    messages: List[Message]
    model: str
    stream: bool
    max_tokens: int


@app.post("/chat/completions")
async def completions(body: Body, request: Request):
    question = body.messages[-1]
    if question.role == 'user':
        question = question.content
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")

    user_question = ''
    print(question)
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
            for response in chatglm_model.chatglm.model.stream_chat(
                    chatglm_model.chatglm.tokenizer,
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
            response, _ = chatglm_model.chatglm.model.chat(
                chatglm_model.chatglm.tokenizer,
                question,
                history,
                max_length=max(2048, body.max_tokens)
            )
            yield json.dumps({"response": response[0]})
        torch_gc()

    async def eval_prompt(model_name):

        def get_question():
            q = question
            if model_name.find('chatglm') != -1:
                q = chat2text(question)

            return zh2en(q)

        q = get_question()
        if model_name.find('gpt2') != -1:
            yield json.dumps(
                {"response": generate_prompt(plain_text=q, max_length=body.max_tokens, model_name="gpt2_650k")}
            )
        elif model_name.find('mj') != -1:
            yield json.dumps({"response": generate_prompt(plain_text=q, max_length=body.max_tokens, model_name="mj")})
        else:
            yield json.dumps(
                {"response": generate_prompt(plain_text=q, max_length=body.max_tokens)}
            )
        torch_gc()

    if body.model.startswith('prompt'):
        return EventSourceResponse(eval_prompt(body.model))

    return EventSourceResponse(eval_chatglm())


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, app_dir=".")
