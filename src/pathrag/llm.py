import base64
import copy
import json
import os
import re
import struct
from functools import lru_cache
from typing import List, Dict, Callable, Any, Union, Optional
import aioboto3
import aiohttp
import numpy as np
import torch
import time
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)

import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url="https://api.openai.com/v1",
    api_key="",
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    time.sleep(2)
    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})


    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug("Full context:")
    if "response_format" in kwargs:
        response = await openai_async_client.beta.chat.completions.parse(
            model=model, messages=messages, **kwargs
        )
    else:
        response = await openai_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    if hasattr(response, "__aiter__"):

        async def inner():
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content is None:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content

        return inner()
    else:
        content = response.choices[0].message.content
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    api_version=None,
    **kwargs,
):
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    if api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    content = response.choices[0].message.content

    return content


class BedrockError(Exception):
    """Generic error for issues related to Amazon Bedrock"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((BedrockError)),
)
async def bedrock_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    **kwargs,
) -> str:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
    )
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
        "AWS_SESSION_TOKEN", aws_session_token
    )
    kwargs.pop("hashing_kv", None)

    messages = []
    for history_message in history_messages:
        message = copy.copy(history_message)
        message["content"] = [{"text": message["content"]}]
        messages.append(message)


    messages.append({"role": "user", "content": [{"text": prompt}]})


    args = {"modelId": model, "messages": messages}


    if system_prompt:
        args["system"] = [{"text": system_prompt}]


    inference_params_map = {
        "max_tokens": "maxTokens",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
    }
    if inference_params := list(
        set(kwargs) & set(["max_tokens", "temperature", "top_p", "stop_sequences"])
    ):
        args["inferenceConfig"] = {}
        for param in inference_params:
            args["inferenceConfig"][inference_params_map.get(param, param)] = (
                kwargs.pop(param)
            )


    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        try:
            response = await bedrock_async_client.converse(**args, **kwargs)
        except Exception as e:
            raise BedrockError(e)

    return response["output"]["message"]["content"][0]["text"]


@lru_cache(maxsize=1)
def initialize_hf_model(model_name):
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    return hf_model, hf_tokenizer


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def hf_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> str:
    model_name = model
    hf_model, hf_tokenizer = initialize_hf_model(model_name)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    kwargs.pop("hashing_kv", None)
    input_prompt = ""
    try:
        input_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]["role"] == "system":
                messages[1]["content"] = (
                    "<system>"
                    + messages[0]["content"]
                    + "</system>\n"
                    + messages[1]["content"]
                )
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for msgid in range(len_message):
                input_prompt = (
                    input_prompt
                    + "<"
                    + ori_message[msgid]["role"]
                    + ">"
                    + ori_message[msgid]["content"]
                    + "</"
                    + ori_message[msgid]["role"]
                    + ">\n"
                )

    input_ids = hf_tokenizer(
        input_prompt, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    output = hf_model.generate(
        **input_ids, max_new_tokens=512, num_return_sequences=1, early_stopping=True
    )
    response_text = hf_tokenizer.decode(
        output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    return response_text




@lru_cache(maxsize=1)
def initialize_lmdeploy_pipeline(
    model,
    tp=1,
    chat_template=None,
    log_level="WARNING",
    model_format="hf",
    quant_policy=0,
):
    from lmdeploy import pipeline, ChatTemplateConfig, TurbomindEngineConfig

    lmdeploy_pipe = pipeline(
        model_path=model,
        backend_config=TurbomindEngineConfig(
            tp=tp, model_format=model_format, quant_policy=quant_policy
        ),
        chat_template_config=(
            ChatTemplateConfig(model_name=chat_template) if chat_template else None
        ),
        log_level="WARNING",
    )
    return lmdeploy_pipe


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def lmdeploy_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    chat_template=None,
    model_format="hf",
    quant_policy=0,
    **kwargs,
) -> str:
    """
    Args:
        model (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download
                        from ii) and iii).
                    - ii) The model_id of a lmdeploy-quantized model hosted
                        inside a model repo on huggingface.co, such as
                        "InternLM/internlm-chat-20b-4bit",
                        "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        chat_template (str): needed when model is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on,
            and when the model name of local path did not match the original model name in HF.
        tp (int): tensor parallel
        prompt (Union[str, List[str]]): input texts to be completed.
        do_preprocess (bool): whether pre-process the messages. Default to
            True, which means chat_template will be applied.
        skip_special_tokens (bool): Whether or not to remove special tokens
            in the decoding. Default to be True.
        do_sample (bool): Whether or not to use sampling, use greedy decoding otherwise.
            Default to be False, which means greedy decoding will be applied.
    """
    try:
        import lmdeploy
        from lmdeploy import version_info, GenerationConfig
    except Exception:
        raise ImportError("Please install lmdeploy before initialize lmdeploy backend.")
    kwargs.pop("hashing_kv", None)
    kwargs.pop("response_format", None)
    max_new_tokens = kwargs.pop("max_tokens", 512)
    tp = kwargs.pop("tp", 1)
    skip_special_tokens = kwargs.pop("skip_special_tokens", True)
    do_preprocess = kwargs.pop("do_preprocess", True)
    do_sample = kwargs.pop("do_sample", False)
    gen_params = kwargs

    version = version_info
    if do_sample is not None and version < (0, 6, 0):
        raise RuntimeError(
            "`do_sample` parameter is not supported by lmdeploy until "
            f"v0.6.0, but currently using lmdeloy {lmdeploy.__version__}"
        )
    else:
        do_sample = True
        gen_params.update(do_sample=do_sample)

    lmdeploy_pipe = initialize_lmdeploy_pipeline(
        model=model,
        tp=tp,
        chat_template=chat_template,
        model_format=model_format,
        quant_policy=quant_policy,
        log_level="WARNING",
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    gen_config = GenerationConfig(
        skip_special_tokens=skip_special_tokens,
        max_new_tokens=max_new_tokens,
        **gen_params,
    )

    response = ""
    async for res in lmdeploy_pipe.generate(
        messages,
        gen_config=gen_config,
        do_preprocess=do_preprocess,
        stream_response=False,
        session_id=1,
    ):
        response += res.response
    return response


async def vllm_model_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
):
    from src.pathrag.vllm_adapter import vllm_model_complete
    
    model = kwargs.pop("model", None)
    if model is None:
        raise ValueError("model must be provided")

    return await vllm_model_complete(
        prompt=prompt,
        model_alias=model,
        system_prompt=system_prompt,
        **kwargs,
    )


async def vllm_embedding(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    from src.pathrag.vllm_adapter import vllm_embed
    
    return await vllm_embed(texts=texts, model_alias=embed_model, **kwargs)


async def vllm_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    from src.pathrag.vllm_adapter import vllm_embed
    
    return await vllm_embed(texts=texts, model_alias=embed_model, **kwargs)


class Model(BaseModel):
    """
    This is a Pydantic model class named 'Model' that is used to define a custom language model.

    Attributes:
        gen_func (Callable[[Any], str]): A callable function that generates the response from the language model.
            The function should take any argument and return a string.
        kwargs (Dict[str, Any]): A dictionary that contains the arguments to pass to the callable function.
            This could include parameters such as the model name, API key, etc.

    Example usage:
        Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]})

    In this example, 'openai_complete_if_cache' is the callable function that generates the response from the OpenAI model.
    The 'kwargs' dictionary contains the model name and API key to be passed to the function.
    """

    gen_func: Callable[[Any], str] = Field(
        ...,
        description="A function that generates the response from the llm. The response must be a string",
    )
    kwargs: Dict[str, Any] = Field(
        ...,
        description="The arguments to pass to the callable function. Eg. the api key, model name, etc",
    )

    class Config:
        arbitrary_types_allowed = True


class MultiModel:
    """
    Distributes the load across multiple language models. Useful for circumventing low rate limits with certain api providers especially if you are on the free tier.
    Could also be used for spliting across diffrent models or providers.

    Attributes:
        models (List[Model]): A list of language models to be used.

    Usage example:
        ```python
        models = [
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_2"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_3"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_4"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_5"]}),
        ]
        multi_model = MultiModel(models)
        rag = LightRAG(
            llm_model_func=multi_model.llm_model_func
            / ..other args
            )
        ```
    """

    def __init__(self, models: List[Model]):
        self._models = models
        self._current_model = 0

    def _next_model(self):
        self._current_model = (self._current_model + 1) % len(self._models)
        return self._models[self._current_model]

    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        kwargs.pop("model", None)   
        kwargs.pop("keyword_extraction", None)
        kwargs.pop("mode", None)
        next_model = self._next_model()
        args = dict(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
            **next_model.kwargs,
        )

        return await next_model.gen_func(**args)


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete("How are you?")
        print(result)

    asyncio.run(main())
