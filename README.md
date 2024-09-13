# llm-api-utils
For more details see [source code](./llm_api_utils.py)  
## How to use
### Setup
```bash
pip install openai google-generativeai anthropic
# set environment variables in .env like .env.example
vim .env
# set environment variables
export $(grep -v '^#' .env | xargs)
```

### Example of usage
```python
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
from llm_api_utils import get_llm_response_async
from tenacity import retry, stop_after_attempt, wait_fixed
model_name = 'models/gemini-1.5-pro-001'
@retry(wait=wait_fixed(90), stop=stop_after_attempt(10))
async def main():
    response = await get_llm_response_async(model_name, params, messages)
    print(response)
asyncio.run(main())
print('#######', model_name)
```

## Links
### OpenAI
[Sample code](./call_gpt.py)
- document: https://platform.openai.com/docs/overview  
- library: https://github.com/openai/openai-python  
- models: https://platform.openai.com/docs/models  
- playground: https://platform.openai.com/playground/chat?models=gpt-4o  
- pricing: https://openai.com/api/pricing/  
- status: https://status.openai.com/  
- cookbook: https://github.com/openai/openai-cookbook
### Anthropic
[Sample code](./call_claude.py)  
- document: https://docs.anthropic.com/en/docs  
- library: https://github.com/anthropics/anthropic-sdk-python  
- models: https://docs.anthropic.com/en/docs/models-overview  
- playground: https://console.anthropic.com/workbench  
- pricing: https://www.anthropic.com/pricing#anthropic-api  
- status: https://status.anthropic.com/  
- cookbook: https://github.com/anthropics/anthropic-cookbook  
### Google
[Sample code](./call_gemini.py)
- document: https://ai.google.dev/gemini-api/docs  
- library: https://github.com/google-gemini/generative-ai-python/tree/main  
- models: https://ai.google.dev/gemini-api/docs/models/gemini  
- playground: https://aistudio.google.com/app/prompts/new_chat  
- pricing: https://ai.google.dev/pricing  
- status: 
- cookbook: https://github.com/google-gemini/cookbook  

## Cost (Input / Output per 1M tokens) 
| Model                     | Input   | Output        |
|---------------------------|-------|------------|
| o1-preview-2024-09-12     | $15.00 | $60.00     |
| gpt-4o-2024-05-13         | $5.00  | $15.00     |
| claude-3-5-sonnet-20240620| $3.00  | $15.00     |
| gemini-1.5-pro-001        | $3.50  | $10.50     |
| o1-mini-2024-09-12        | $3.00  | $12.00     |
| gpt-4o-mini-2024-07-18    | $0.15  | $0.60      |
| gemini-1.5-flash-001      | $0.075 | $0.30      |



## RPM / TPM
| Model  | RPM    | TPM        |
|--------|--------|------------|
| gpt-4o-2024-05-13 [Tier 5](https://platform.openai.com/docs/guides/rate-limits/tier-5-rate-limits)| 10,000 | 30,000,000 |
| claude-3-5-sonnet-20240620 [Tier 4](https://docs.anthropic.com/en/api/rate-limits#rate-limits)| 4,000  | 400,000    |
| gemini-1.5-pro-001 [Pay-as-you-go](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro)| 360  | 4,000,000  |
| gpt-4o-mini-2024-07-18 [Tier 5](https://platform.openai.com/docs/guides/rate-limits/tier-5-rate-limits)| 30,000 | 150,000,000 |
| gemini-1.5-flash-001 [Pay-as-you-go](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash)| 1,000  | 4,000,000  |
| o1-preview-2024-09-12 [Tier 5](https://platform.openai.com/docs/guides/rate-limits/tier-5-rate-limits)| 20 | 30,000,000 |
| o1-mini-2024-09-12 [Tier 5](https://platform.openai.com/docs/guides/rate-limits/tier-5-rate-limits)| 20 | 150,000,000 |