import google.generativeai as genai
import os
import asyncio


# genai.configure()

model_name = 'gemini-1.5-flash-001'
generation_config = {
    "temperature": 0.0,
    "max_output_tokens": 256,
    "response_mime_type": "text/plain",
}

client = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config,
)

messages = [
    {"role": "user", "parts": ["大喜利しましょう。とても面白い回答をしてくださいね。"]},
    {"role": "model", "parts": ["おけ、任せて\n"]},
    {"role": "user", "parts": ["こんな台風は嫌だ、どんな台風？"]},
]

response = client.generate_content(messages)

# print(response)
print(repr(response.text))
# print(response.candidates[0].content)

from utils import retry_with_linear_backoff

# error around async, we have to make client instance inside the function
# https://github.com/search?q=repo%3Agoogle-gemini%2Fgenerative-ai-python+generate_content_async%28&type=issues

@retry_with_linear_backoff(delay=60, max_retries=5)
async def main() -> None:
    client = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )
    response = await client.generate_content_async(
        messages,
    )
    print(repr(response.text))
asyncio.run(main())

import json
import requests
import aiohttp

@retry_with_linear_backoff(delay=60, max_retries=5)
async def get_response(messages):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    params = {
        "key": os.environ["GOOGLE_API_KEY"], # unsafe?
    }
    headers = {
        'Content-Type': 'application/json',
    }
    contents = []
    for message in messages:
        contents.append(
            {
                "role": message["role"],
                "parts": [{"text": message["parts"][0]}]
            }
        )
    data = {"contents": contents, "generationConfig": generation_config}
    # response = await requests.post(url, headers=headers, data=json.dumps(data))
    # response = response.json()
    # print(response)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data), params=params) as response:
            response_data = await response.json()
            print(response_data)
asyncio.run(get_response(messages))
