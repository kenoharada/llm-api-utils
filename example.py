from llm_api_utils import get_llm_response


model_name = 'gpt-4o-mini-2024-07-18'
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

print(messages[-1]['content'])
response = get_llm_response(model_name, params, messages)
print(response)
print('#######', model_name)

model_name = 'claude-3-5-sonnet-20240620'
response = get_llm_response(model_name, params, messages)
print(response)
print('#######', model_name)

# example of asynchronous request
import asyncio
from llm_api_utils import get_llm_response_async, retry_with_linear_backoff
model_name = 'models/gemini-1.5-flash-001'
@retry_with_linear_backoff(delay=60, max_retries=5)
async def main():
    response = await get_llm_response_async(model_name, params, messages)
    print(response)
asyncio.run(main())
print('#######', model_name)