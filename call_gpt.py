from openai import OpenAI
import asyncio


client = OpenAI()
model_name = 'gpt-4o-2024-05-13'
max_tokens = 4096
temperature = 0.0
messages = [
    {"role": "user", "content": "大喜利しましょう。とても面白い回答をしてくださいね。"},
    {"role": "assistant", "content": "おけ、任せて"},
    {"role": "user", "content": "こんな台風は嫌だ、どんな台風？"}
]

response = client.chat.completions.create(
    max_tokens=max_tokens,
    messages=messages,
    temperature=temperature,
    model=model_name,
)
print(response.choices[0].message.content)

from utils import retry_with_linear_backoff
from openai import AsyncOpenAI
client = AsyncOpenAI()

@retry_with_linear_backoff(delay=60, max_retries=5)
async def main() -> None:
    response = await client.chat.completions.create(
        max_tokens=max_tokens,
        messages=messages,
        temperature=temperature,
        model=model_name,
    )
    print(response.choices[0].message.content)
asyncio.run(main())
