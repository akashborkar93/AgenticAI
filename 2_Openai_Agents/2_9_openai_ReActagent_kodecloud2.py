import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tools=[
    {
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "Check the calendar for a given day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "string", "description": "Day of the week"}
                },
                "required": ["day"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_contact",
            "description": "Look up a contact's email address by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description":"The contact name" }
                },
                "required":["name"]
            }
        }
    }
]

def check_calendar(day):
    events= {
        "monday": "Team standup at 9am",
        "tuesday": "Dentist at 2pm",
        "wednesday": "No events",
        "thursday": "1pm lunch with Alex, 3pm product review",
        "friday": "Weekly retro at 4pm"
    }
    return events.get(day.lower(), "No events found for that day.")

def search_contact(name):
    contacts = {
        "sarah": "sarah@email.com",
        "alex": "alex@email.com",
        "jordan": "jordan@email.com"
    }
    return contacts.get(name.lower(), "No contacts found")

user_message = "What's on my calendar today? Also find Sarah's email and send me a summary."

def run_agent(system_prompt):
    messages = [
        {
            "role": "system", "content": system_prompt
        },
        {
            "role": "user", "content": user_message
        }
    ]
    while True:
        response= client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                if tc.function.name == "check_calendar":
                    result = check_calendar(**args)
                elif tc.function.name == "search_contact":
                    result = search_contact(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
        else:
            print(msg.content)
            break



print("--- No ReAct ---")
run_agent("You are a helpful personal assistant.")

print("\n--- ReAct ---")
run_agent("You are a helpful personal assistant. Before every tool call, write 'Thought: [your reasoning]'. After every tool result, write 'Observation: [what you learned]'. Then decide your next step.")
