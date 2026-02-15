# pip install openai python-dotenv

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

openai = OpenAI(
    base_url="https://kodekey.ai.kodekloud.com/v1",
    api_key=os.getenv("KODEKLOUD_API_KEY")
)

'''
# First a very basic question
messages_created = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant who explains answers clearly and concisely."
    },
    {
        "role": "user",
        "content": "What is the factorial of 7?"
    },
    {
        "role": "assistant",
        "content": "The factorial of 7 is 5040."
    },
    {
        "role": "user",
        "content": "Now explain how factorial works in simple terms."
    }
]

response = openai.chat.completions.create(
    model="openai/gpt-4o-2024-11-20",
    messages=messages_created,
    temperature=0.3
)

# print(response)
print(response.choices[0].message.content)

'''


#Now let us ask a tougher question
question = " Provide me list of devops tools Respond only with the question in one line."
messages = [{"role": "user", "content": question}]

response = openai.chat.completions.create(
    model="openai/gpt-4o-2024-11-20",
    messages=messages
)

# Save the question returned by the LLM into a variable called question
question = response.choices[0].message.content

print(f"OpenAI Question: {question}")  

# now form a new message list
messages = [{
             "role": "user", 
			 "content": question
		   }]

response = openai.chat.completions.create(
    model="openai/gpt-4o-2024-11-20",
    messages=messages
)

answer = response.choices[0].message.content
print(f"GPT answer: {answer} ")

