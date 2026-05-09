import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "Check the user's calendar for events on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check in YYYY-MM-DD format."
                    }
                },
                "required": ["date"]
            }
        }
    }
]

def check_calendar(date):
    return "10am: Team standup, 2pm: Dentist appointment"

def execute_tool(name, arg):
    if name=="check_calendar":
        return check_calendar(arg)
    else:
        return f"unknown tool: {name}"



system_message = "You are a helpful personal assistant."

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "What's on my calendar today?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)

finish_reason = response.choices[0].finish_reason
print(finish_reason)

if finish_reason == "tool_calls":
    assistent_message = response.choices[0].message
    messages.append(assistent_message)

    for tool_call in assistent_message.tool_calls:
        name = tool_call.function.name
        arg = json.loads(tool_call.function.arguments)
        result = execute_tool(name, arg)

        messages.append({
            "role": "tool",
            "tool_call_id":tool_call.id,
            "content": result
        
        })

        final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        
        )
        print(final_response.choices[0].message.content)

else:
     print(response.choices[0].message.content)


