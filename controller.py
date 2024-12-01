from llm import LLM
from dotenv import load_dotenv
from os import getenv
from segmentor import ImageSegmenter
import re

load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")

class Controller:
	def __init__(self, general_base_prompt_file: str = './prompts/general_model_prompt.txt', execution_base_prompt_file: str = './prompts/execution_model_prompt.txt') -> None:
		self.general_llm = LLM(api_key=OPENAI_API_KEY, temperature=0.5, max_tokens=1000)
		self.execution_llm = LLM(api_key=OPENAI_API_KEY, temperature=0.5, max_tokens=1000)
		self.general_base_prompt_file = open(general_base_prompt_file, "r").read()
		self.execution_base_prompt = open(execution_base_prompt_file, "r").read()

		self.function_mapping = {

		}

		self.actions: list[dict] = [] # Contains dictionaries with the title, function_name, arguments and result of each action
		self.num_actions: int = 0

	def run(self, prompt: str, image_path) -> str:
		self.general_llm.clear_history()
		self.execution_llm.clear_history()

		# Get response from general LLM
		general_response = self.general_llm.ask(self.general_base_prompt_file + prompt)
		filtered_general_response = self.filter_general_response(general_response)
		goal, action_titles = self.split_general_actions(filtered_general_response)

		for action_title in action_titles:
			execution_response = self.execution_llm.ask(self.execution_base_prompt + prompt)
			function_call_string = self.parse_function_call(execution_response)
			self.actions.append({
				"title": action_title,
				"function_name": function_call_string["function_name"],
				"arguments": function_call_string["arguments"],
			})
			self.num_actions += 1

		final_answer = self.execution_llm.ask(self.execution_base_prompt + prompt)

		return self.parse_final_answer(execution_response)
	
	@staticmethod
	def filter_general_response(general_response: str) -> str:
		# Filter out the general response by considering only the text after "Tasks:"
		keyword = "Tasks:"
		tasks_index = general_response.lower().find(keyword.lower())

		if tasks_index == -1:
			return general_response
		
		else:
			return general_response[tasks_index + len(keyword):].strip()
		
	@staticmethod
	def parse_function_call(execution_output: str) -> dict:
		"""
		Parses the execution message, extracts the function call, and executes the function.

		Parameters:
			message (str): The complete execution message.

		Returns:
			dict: A dictionary containing:
				- 'action_description' (str): The description after "Next Action:".
				- 'function_result' (str): The result of the function execution.
		"""
		# Split the message into paragraphs
		paragraphs = execution_output.strip().split("\n\n")
		
		# Ensure the message has at least three paragraphs
		if len(paragraphs) < 3:
			raise ValueError("The message format is invalid or incomplete.")
		
		# Extract the first paragraph and process "Next Action"
		action_description = paragraphs[0].replace("Next Action: ", "").strip()
		
		# Extract the function call from the last paragraph
		function_call = paragraphs[2].replace("Function to Call: ", "").strip()
		
		# Dynamically parse the function name and argument
		function_name, argument = function_call.strip('[]').split("(", 1)
		argument = argument.strip(")").strip("'")
		
		# Execute the function dynamically
		if function_name in self.function_mapping:
			function_result = self.function_mapping[function_name](argument)
		else:
			raise ValueError(f"Function '{function_name}' is not defined.")
		
		return {
			"action_description": action_description,
			"function_result": function_result,
			"function_name": function_name,
		}

	@staticmethod
	def parse_final_answer(execution_output: str) -> str:
		# Extract the final answer from the execution output
		final_answer_index = execution_output.lower().find("final answer:")

		if final_answer_index == -1:
			return execution_output
		else:
			return execution_output[final_answer_index:].strip()

	@staticmethod
	def split_general_actions(general_response: str) -> tuple:
		# Get all the lines that have a [ ] in them
		action_titles = [line[line.find("]") + 1:line.find("]")].strip() for line in general_response.split("\n") if re.search(r"\[.*\]", line)]
		goal = action_titles[0]
		action_titles = action_titles[1:]

		return goal, action_titles

