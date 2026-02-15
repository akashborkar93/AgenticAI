from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(
    base_url="https://kodekey.ai.kodekloud.com/v1",
    api_key=os.getenv("KODEKLOUD_API_KEY")
)

messages = [
    {
        "role": "user",
        "content": "In a single sentence, why should someone choose KodeKloud over other platforms to learn DevOps?"
    }
]

response = client.chat.completions.create(
    model="openai/gpt-4o-2024-11-20",
    messages=messages
)

print(response.choices[0].message.content)