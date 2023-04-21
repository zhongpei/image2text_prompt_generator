import time
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from typing import List

import torch
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel
from transformers import LogitsProcessor, LogitsProcessorList

from config import settings
from .singleton import Singleton

device = "cuda" if torch.cuda.is_available() else "cpu"
if settings.chatglm.device == "cpu":
    device = "cpu"
DEVICE = device
DEVICE_ID = "0" if torch.cuda.is_available() else None
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': 0, 'lm_head': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
    return "".join(lines)


class InvalidScoreLogitsProcessor(LogitsProcessor):

    def __init__(self, start_pos=20005):
        self.start_pos = start_pos

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., self.start_pos] = 5e4
        return scores


class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10

    def predict_continue(self, query, latest_message, max_length, top_p,
                         temperature, allow_generate, history, *args,
                         **kwargs):
        if history is None:
            history = []
        allow_generate[0] = True
        history.append((query, latest_message))
        for response in self.stream_chat_continue(
                self.model,
                self.tokenizer,
                query=query,
                history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature):
            history[-1] = (history[-1][0], response)
            yield history, '', ''
            if not allow_generate[0]:
                break

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token,
            temperature=self.temperature,
        )
        torch_gc()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, response]]
        return response

    def __init__(self):
        super().__init__()
        model_name = settings.chatglm.model

        print(f'Loading model {model_name} on {device}')
        start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=settings.chatglm.local_files_only,
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=settings.chatglm.local_files_only,
        )
        if device == 'cuda' or device == "mps":
            print("\n\n\n\ncuda\n\n\n\n")
            model = model.half().to(device)
        else:
            print("\n\n\n\ncpu\n\n\n\n")
            model = model.to(device).float()
            # model = model.quantize(bits=4, kernel_file=kernel_file)
        model = model.eval()
        self.model = model
        self.model_name = model_name
        end = time.perf_counter()
        print(
            f'Successfully loaded model {model_name}, time cost: {end - start:.2f}s'
        )

    @torch.no_grad()
    def generator_image_text(self, text: str, prompt: str) -> str:
        if prompt is None or len(prompt.strip()) == 0:
            prompt = "描述画面:\"{}\"".format(text)
        else:
            prompt = prompt + "\"{}\"".format(text)

        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response

    @torch.no_grad()
    def stream_chat_continue(self,
                             model,
                             tokenizer,
                             query: str,
                             history: List[Tuple[str, str]] = None,
                             max_length: int = 2048,
                             do_sample=True,
                             top_p=0.7,
                             temperature=0.95,
                             logits_processor=None,
                             **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        if len(history) > 0:
            answer = history[-1][1]
        else:
            answer = ''
        logits_processor.append(
            InvalidScoreLogitsProcessor(
                start_pos=20005 if 'slim' not in self.model_name else 5))
        gen_kwargs = {
            "max_length": max_length,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs
        }
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                if i != len(history) - 1:
                    prompt += "[Round {}]\n问：{}\n答：{}\n".format(
                        i, old_query, response)
                else:
                    prompt += "[Round {}]\n问：{}\n答：".format(i, old_query)
        batch_input = tokenizer([prompt], return_tensors="pt", padding=True)
        batch_input = batch_input.to(model.device)

        batch_answer = tokenizer(answer, return_tensors="pt")
        batch_answer = batch_answer.to(model.device)

        input_length = len(batch_input['input_ids'][0])
        final_input_ids = torch.cat(
            [batch_input['input_ids'],
             batch_answer['input_ids'][:, :-2]],
            dim=-1
        ).to(model.device)

        attention_mask = model.get_masks(
            final_input_ids, device=final_input_ids.device
        )

        batch_input['input_ids'] = final_input_ids
        batch_input['attention_mask'] = attention_mask

        input_ids = final_input_ids
        MASK, gMASK = self.model.config.bos_token_id - 4, self.model.config.bos_token_id - 3
        mask_token = MASK if MASK in input_ids else gMASK
        mask_positions = [seq.tolist().index(mask_token) for seq in input_ids]
        batch_input['position_ids'] = self.model.get_position_ids(
            input_ids, mask_positions, device=input_ids.device
        )

        for outputs in model.stream_generate(**batch_input, **gen_kwargs):
            outputs = outputs.tolist()[0][input_length:]
            response = tokenizer.decode(outputs)
            response = model.process_response(response)
            yield parse_codeblock(response)


@Singleton
class Models(object):

    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)

        if item == 'chatglm':
            self.chatglm = ChatGLM()

        return getattr(self, item)


models = Models.instance()


def chat2text(text: str, prompt: str) -> str:
    return models.chatglm.generator_image_text(text, prompt)
