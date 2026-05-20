from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


response = client.responses.create(
    model="gpt-4o-mini",
    input="Write two lines poem on pallavi."
)

print(response.output_text)