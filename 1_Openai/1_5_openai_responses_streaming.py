from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

stream = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": "Say 'double bubble bath' slowly and repeat it 3 times.",
        },
    ],
    stream=True,
)   

for event in stream:
    print(event)