import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()


SYS_PROMPT = """1-You are Legal Pakistani Ai Assistant that can answer user's questions from your knowledge.
                2-You can use tools binded with yourself
                3-Give Concise Answers without uncessary information
                4-DONT use uncessary symbols like astericks(*) and dashes(-) in conversation
                5-If summarizing something dont use symbols like (*,-,#) to bold things 
"""


def get_groq_llm():
    return ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY_2")
    )
