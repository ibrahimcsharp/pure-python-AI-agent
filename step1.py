from openai import OpenAI

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

client = OpenAI(api_key="")

response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {"role": "user", "content": "Explain what an AI agent is in one sentence."},
    ],
)

print(response.choices[0].message.content)
