from llm import LLM
from dotenv import load_dotenv
from os import getenv

load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")

def user_interaction(image, text: str) -> str:
    # Initialize the LLM instance
    llm = LLM(OPENAI_API_KEY, temperature=0.5, max_tokens=1000)
    
    # Step 1: Ask the LLM to generate a clarification question based on the ambiguous input
    clarification_prompt = f"The following text is unclear: '{text}'. Please generate a clear question to ask the user in order to clarify the meaning of the request."
    clarification_question = llm.ask(clarification_prompt)
    
    # Print and ask the user for a response to the clarification question
    # TODO
    user_response = ''
    
    # Step 2: Analyze the user's response (this will be done automatically by the LLM)
    analysis_prompt = f"Here is the user response to the clarification: '{user_response}'. Is it clear and sufficient to proceed with the task? If yes, summarize the user's answer. If no, make assumptions based on your own understanding of the request."
    
    # Step 3: Return the analysis (whether the response was sufficient or assumptions were made)
    return llm.ask(analysis_prompt), ''


def calculate(image, text: str) -> str:
    llm = LLM(OPENAI_API_KEY, temperature=0.25, max_tokens=250, system_prompt='Your task is given the extra information calculate or abstract the information asked. You have enough resources to do it.')
    return llm.ask(text), ''