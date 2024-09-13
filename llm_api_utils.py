import time
import os
import asyncio
from typing_extensions import get_args
from tenacity import retry, wait_fixed, stop_after_attempt


import openai
from openai import OpenAI, AsyncOpenAI
import anthropic
from anthropic import Anthropic, AsyncAnthropic
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions


def get_llm_response(model_name: str, params_: dict, messages: list[dict]) -> str:
    params = params_.copy()
    if model_name in OPENAI_MODEL_NAMES:
        if 'max_tokens' in params:
            params['maxcompletion_tokens'] = params['max_tokens']
            del params['max_tokens']
            return get_gpt_respnose(model_name, params, messages)
        return get_gpt_respnose(model_name, params, messages)
    elif model_name in ANTHROPIC_MODEL_NAMES:
        if 'max_tokens' not in params:
            params['max_tokens'] = 8192
        return get_claude_response(model_name, params, messages)
    elif model_name in GEMINI_MODEL_NAMES:
        return get_gemini_response(model_name, params, messages)
    else:
        raise ValueError(f"model_name {model_name} not supported. Supported model names are: {OPENAI_MODEL_NAMES + ANTHROPIC_MODEL_NAMES + GEMINI_MODEL_NAMES}")
    

@retry(wait=wait_fixed(90), stop=stop_after_attempt(10))
async def get_llm_response_async(model_name: str, params_: dict, messages: list[dict]) -> str:
    params = params_.copy()
    if model_name in OPENAI_MODEL_NAMES:
        if 'max_tokens' in params:
            params['maxcompletion_tokens'] = params['max_tokens']
            del params['max_tokens']
            return get_gpt_respnose_async(model_name, params, messages)
        return get_gpt_respnose_async(model_name, params, messages)
    elif model_name in ANTHROPIC_MODEL_NAMES:
        if 'max_tokens' not in params:
            params['max_tokens'] = 8192
        return await get_claude_response_async(model_name, params, messages)
    elif model_name in GEMINI_MODEL_NAMES:
        return await get_gemini_response_async(model_name, params, messages)
    else:
        raise ValueError(f"model_name {model_name} not supported. Supported model names are: {OPENAI_MODEL_NAMES + ANTHROPIC_MODEL_NAMES + GEMINI_MODEL_NAMES}")


def get_gpt_respnose(model_name: str, params: dict, messages: list[dict]) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        **params
    )
    return response.choices[0].message.content

async def get_gpt_respnose_async(model_name: str, params: dict, messages: list[dict]) -> str:
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        messages=messages,
        model=model_name,
        **params
    )
    return response.choices[0].message.content


def get_claude_response(model_name: str, params: dict, messages: list[dict]) -> str:
    client = Anthropic()
    if messages[0]['role'] == 'system':
        response = client.messages.create(
            messages=messages[1:],
            model=model_name,
            system=messages[0]['content'],
            **params
        )
    else:
        response = client.messages.create(
            messages=messages,
            model=model_name,
            **params
        )
    return response.content[0].text

async def get_claude_response_async(model_name: str, params: dict, messages: list[dict]) -> str:
    client = AsyncAnthropic()
    if messages[0]['role'] == 'system':
        response = await client.messages.create(
            messages=messages[1:],
            model=model_name,
            system=messages[0]['content'],
            **params
        )
    else:
        response = await client.messages.create(
            messages=messages,
            model=model_name,
            **params
        )
    return response.content[0].text

from google.generativeai.types import HarmCategory, HarmBlockThreshold
def get_gemini_response(model_name: str, params: dict, messages: list[dict]) -> str:
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    generation_config = parse_gemini_generation_config(params)
    if messages[0]['role'] == 'system':
        client = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=messages[0]['content'],
        )
    else:
        client = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )
    gemini_messages = parse_gemini_messages(messages)
    response = client.generate_content(gemini_messages, safety_settings=safety_settings)
    return response.text

async def get_gemini_response_async(model_name: str, params: dict, messages: list[dict]) -> str:
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    generation_config = parse_gemini_generation_config(params)
    if messages[0]['role'] == 'system':
        client = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=messages[0]['content'],
        )
    else:
        client = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )
    gemini_messages = parse_gemini_messages(messages)
    try:
        response = await client.generate_content_async(gemini_messages, safety_settings=safety_settings)
        return response.text
    except ValueError as e:
        # https://github.com/google-gemini/generative-ai-python/issues/282
        print(e)
        print(response)
        return ''


def parse_gemini_generation_config(params: dict) -> dict:
    generation_config = dict()
    for param_key in params:
        if param_key == 'max_tokens':
            generation_config['max_output_tokens'] = params[param_key]
        else:
            generation_config[param_key] = params[param_key]
    return generation_config


def parse_gemini_messages(messages: list[dict]) -> list[dict]:
    gemini_messages = []
    for message in messages:
        gemini_message = dict()
        if message['role'] == 'user':
            role = 'user'
        elif message['role'] == 'assistant':
            role = 'model'
        elif message['role'] == 'system':
            continue
        else:
            pass

        gemini_message['role'] = role
        if 'parts' in message:
            gemini_message['parts'] = message['parts']
        else:
            gemini_message['parts'] = [message['content']+'\n']
        gemini_messages.append(gemini_message)
    return gemini_messages


def get_gpt_model_names():
    client = OpenAI()
    # https://platform.openai.com/docs/api-reference/models/list
    openai_model_names = [model_info.id for model_info in client.models.list().data]
    return openai_model_names


def get_gemini_model_names():
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini_model_names = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            gemini_model_names.append(m.name)
    return gemini_model_names


def get_anthropic_model_names():
    # https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/model.py
    model_names = [model_name for model_name in get_args(get_args(anthropic.types.model.Model)[-1])]
    return model_names


OPENAI_MODEL_NAMES = get_gpt_model_names()
GEMINI_MODEL_NAMES = get_gemini_model_names()
ANTHROPIC_MODEL_NAMES = get_anthropic_model_names()


if __name__ == '__main__':
    # model_name = 'gpt-4o-mini-2024-07-18'
    # model_name = 'models/gemini-1.5-flash-001'
    model_name = 'claude-3-5-sonnet-20240620'
    params = {
        'max_tokens': 256, 
        'temperature': 0.0
    }
    messages = [
        {"role": "system", "content": "回答の際は、3つの回答を箇条書きで回答してください。"},
        {"role": "user", "content": "大喜利しましょう。とても面白い回答をしてくださいね。"},
        {"role": "assistant", "content": "おけ、任せて"},
        {"role": "user", "content": "こんな台風は嫌だ、どんな台風？"}
    ]
    response = get_llm_response(model_name, params, messages)
    print(response)

    async def main():
        response = await get_llm_response_async(model_name, params, messages)
        print(response)
    asyncio.run(main())