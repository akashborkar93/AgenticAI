from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, function_tool

load_dotenv(override=True)

# Enable OpenAI tracing (shows in https://platform.openai.com/logs?api=traces)
client = OpenAI()

knowledge_base = {
    "shipping time": "Our standard shipping time is 3-5 business days.",
    "return policy": "You can return any product within 30 days of delivery.",
    "warranty": "All products come with a one-year warranty covering manufacturing defects.",
    "payment methods": "We accept credit cards, debit cards, and PayPal.",
    "customer support": "You can reach our support team 24/7 via email or chat."
}

@function_tool
async def faq_invoker(topic: str) -> str:
    """Provides answers to frequently asked customer support questions."""
    user_query = topic.lower()
    for topic_key, answer in knowledge_base.items():
        if topic_key in user_query:
            return answer

    fallback = (
        "I'm sorry, but I couldn't find specific information about that topic. "
        "Please check the company's website or contact customer support directly."
    )
    return fallback

faq_agent = Agent(
    name="Customer Support Bot",
    instructions=(
        "You are a customer support assistant. "
        "You MUST ALWAYS use the faq_invoker tool to answer questions. "
        "DO NOT answer from your own knowledge. "
        "Use the tool even if you think you know the answer."
    ),
    tools=[faq_invoker]
)

async def chat_with_support(message):
    session = await Runner.run(faq_agent, message)
    return session.final_output

async def main():
    print("Customer Support Bot is running. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting.")
            break
        response = await chat_with_support(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    asyncio.run(main())