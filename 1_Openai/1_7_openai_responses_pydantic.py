from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)

client = OpenAI()

class CalenderEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-4o-mini",
    input=[
        {
            "role": "system",
            "content": "Extract event information"},
        {
            "role": "user",
            "content": "Alice and Bob are going to a picnic on Sunday." 
        },
    ],
    text_format=CalenderEvent,
)

event = response.output_parsed

print("Parsed Event (Pydantic object):", event)
print("Event name:", event.name)
print("Event date:", event.date)
print("Participants:", ", ".join(event.participants))