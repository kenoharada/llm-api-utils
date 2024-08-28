from anthropic import Anthropic
import asyncio


client = Anthropic(max_retries=5)
messages = [
    {"role": "user", "content": "大喜利しましょう。とても面白い回答をしてくださいね。"},
    {"role": "assistant", "content": "おけ、任せて"},
    {"role": "user", "content": "こんな台風は嫌だ、どんな台風？"}
]

message = client.messages.create(
    max_tokens=8192,
    messages=messages,
    temperature=0,
    model="claude-3-5-sonnet-20240620",
)
print(message.content[0].text)
print(message.usage)

messages_text = ''.join([m['content'] for m in messages])
token_count = client.count_tokens(messages_text)
print('Input Token count(by older tokenizer):', token_count)
print('Input Token count(by newer tokenizer):', message.usage.input_tokens)
print('Input Token count difference:', token_count - message.usage.input_tokens)

token_count = client.count_tokens(message.content[0].text)
print('Output Token count(by older tokenizer):', token_count)
print('Output Token count(by newer tokenizer):', message.usage.output_tokens)
print('Output Token count difference:', token_count - message.usage.output_tokens)

# from anthropic import AsyncAnthropic
# client = AsyncAnthropic(max_retries=5)

# async def main() -> None:
#     message = await client.messages.create(
#         max_tokens=8192,
#         messages=messages,
#         temperature=0,
#         model="claude-3-5-sonnet-20240620",
#     )
#     print(message.content[0].text)
#     print(message.usage)
# asyncio.run(main())
