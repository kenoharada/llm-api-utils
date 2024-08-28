# llm-api-utils

## Links
### OpenAI
- document: https://platform.openai.com/docs/overview  
- library: https://github.com/openai/openai-python  
- models: https://platform.openai.com/docs/models  
- playground: https://platform.openai.com/playground/chat?models=gpt-4o  
- pricing: https://openai.com/api/pricing/  
- status: https://status.openai.com/  
### Anthropic
[Sample code](./call_claude.py)  
- document: https://docs.anthropic.com/en/docs  
- library: https://github.com/anthropics/anthropic-sdk-python  
- models: https://docs.anthropic.com/en/docs/models-overview  
- playground: https://console.anthropic.com/workbench  
- pricing: https://www.anthropic.com/pricing#anthropic-api  
- status: https://status.anthropic.com/
### Google
- pricing: https://ai.google.dev/pricing

## Cost (Input / Output per 1M tokens) 
| Model                   | Input Cost | Output Cost |
|-------------------------|------------|-------------|
| gpt-4o-2024-05-13       | $5.00      | $15.00      |
| claude-3-5-sonnet-20240620 | $3.00      | $15.00      |
| gemini-1.5-pro-001      | $3.50      | $10.50      |
| gpt-4o-mini-2024-07-18  | $0.15      | $0.60       |
| gemini-1.5-flash-001    | $0.075     | $0.30       |



## RPM / TPM
| Model  | RPM    | TPM        |
|--------|--------|------------|
| gpt-4o-2024-05-13 [Tier 5](https://platform.openai.com/docs/guides/rate-limits/tier-5-rate-limits)| 10,000 | 30,000,000 |
| claude-3-5-sonnet-20240620 [Tier 4](https://docs.anthropic.com/en/api/rate-limits#rate-limits)| 4,000  | 400,000    |
| gemini-1.5-pro-001 [Pay-as-you-go](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro)| 360  | 4,000,000  |
| gpt-4o-2024-05-13 [Tier 5](https://platform.openai.com/docs/guides/rate-limits/tier-5-rate-limits)| 30,000 | 150,000,000 |
| gemini-1.5-flash-001 [Pay-as-you-go](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash)| 1,000  | 4,000,000  |
