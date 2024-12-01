from llm import LLM
from dotenv import load_dotenv
from os import getenv

load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")

def calculate(image, text: str) -> str:
    llm = LLM(OPENAI_API_KEY, temperature=0.25, max_tokens=250, system_prompt='Your task is given the extra information calculate or abstract the information asked. You have enough resources to do it.')
    return llm.ask(text), ''