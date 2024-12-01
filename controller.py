from llm import LLM
from dotenv import load_dotenv
from os import getenv
from segmentor import ImageSegmenter
import re
from functions import calculate

load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")

class Controller:
	def __init__(self, general_base_prompt_file: str = './prompts/general_model_prompt.txt', execution_base_prompt_file: str = './prompts/execution_model_prompt.txt') -> None:
		self.general_llm = LLM(api_key=OPENAI_API_KEY, temperature=0.5, max_tokens=1000)
		self.execution_llm = LLM(api_key=OPENAI_API_KEY, temperature=0.25, max_tokens=1000)
		self.general_base_prompt_file = open(general_base_prompt_file, "r").read()
		self.execution_base_prompt = open(execution_base_prompt_file, "r").read()


		import sys
		import os
		current_dir = os.getcwd()
		print(current_dir)
		sys.path.append(os.path.abspath(os.path.join(current_dir, '/samsam/')))
		from sam2.build_sam import build_sam2
		from sam2.sam2_image_predictor import SAM2ImagePredictor
		import numpy as np
		import torch
		import cv2
		from PIL import Image
		import matplotlib.pyplot as plt
		from hydra import initialize_config_module
		from hydra.core.global_hydra import GlobalHydra
		if not GlobalHydra.instance().is_initialized():
			initialize_config_module("samsam/sam2", version_base="1.2")


		# self.segmenter = ImageSegmenter(
		# 	grounding_dino_cfg="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
		# 	grounding_dino_weights="GroundingDINO/weights/groundingdino_swint_ogc.pth",
		# 	sam_cfg="C:/Users/Usuario/Documents/Projectes/LauzHack2024/LauzHack2024/samsam/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
		# 	sam_weights="C:/Users/Usuario/Documents/Projectes/LauzHack2024/LauzHack2024/samsam/weights/sam2.1_hiera_tiny.pt"
		# )

		self.segmenter = ImageSegmenter(
			grounding_dino_cfg="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
			grounding_dino_weights="GroundingDINO/weights/groundingdino_swint_ogc.pth",
			sam_cfg="C:/Users/Cai Selvas Sala/GIA_UPC/Personal/LauzHack/LauzHack_2024/LauzHack2024/samsam/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
			sam_weights="C:/Users/Cai Selvas Sala/GIA_UPC/Personal/LauzHack/LauzHack_2024/LauzHack2024/samsam/weights/sam2.1_hiera_tiny.pt"
		)
		self.function_mapping = {
			"segment": self.segmenter.segment,
			"calculate": calculate,
			"segment_unique": self.segmenter.segment_unique,
			"count_people": self.segmenter.count_people,
			"change_color": self.segmenter.change_color,
			"calculate_crop_percentage": self.segmenter.calculate_crop_percentage
		}

		self.actions: list[dict] = [] # Contains dictionaries with the title, function_name, arguments and result of each action
		self.num_actions: int = 0

	def run(self, prompt: str, image_path: str) -> str:
		self.general_llm.clear_history()
		self.execution_llm.clear_history()

		# Get response from general LLM
		general_response = self.general_llm.ask(self.general_base_prompt_file + prompt)
		filtered_general_response = self.filter_general_response(general_response)
		goal, action_titles = self.split_general_actions(filtered_general_response)

		for i, action_title in enumerate(action_titles):
			execution_response = self.execution_llm.ask(self.concatenate_execution_prompt(action_title, goal, [self.actions[j]["result"] for j in range(i)]))
			function_call_string = self.parse_function_call(execution_response)
			print(function_call_string)
			result, image_path = self.execute_function(function_call_string, image_path)
			self.actions.append({
				"title": action_title,
				"function_name": function_call_string["function_name"],
				"arguments": function_call_string["arguments"],
				"answer": function_call_string["answer"],
				"result": result,
				"image_path": image_path
			})
			self.num_actions += 1

			results_message = f"Task: {action_title}\nResult: {result}"
			print(results_message)
			
		final_answer = self.execution_llm.ask(self.concatenate_execution_prompt("Now with all the information you must answer the question in order to achieve the GOAL.", goal, [self.actions[j]["result"] for j in range(self.num_actions)]))

		# return self.parse_final_answer(final_answer)
		dictionary = self.parse_function_call(final_answer)
		
		return dictionary["answer"]
	
	@staticmethod
	def filter_general_response(general_response: str) -> str:
		# Filter out the general response by considering only the text after "Tasks:"
		keyword = "Tasks:"
		tasks_index = general_response.lower().find(keyword.lower())

		if tasks_index == -1:
			return general_response
		
		else:
			return general_response[tasks_index + len(keyword):].strip()
		
	def parse_function_call(self, execution_output: str) -> dict:
		function_name, arguments, answer = None, None, None

		function_to_call_index = execution_output.lower().find("function to call:")
		answer_index = execution_output.lower().find("answer:")

		if (function_to_call_index == -1) and (answer_index == -1):
			return {
				"function_name": function_name,
				"arguments": arguments,
				"answer": answer
			}

		answer = execution_output[answer_index + 1: function_to_call_index].strip()

		if function_to_call_index != -1:
			function_call_line = execution_output[function_to_call_index + len("Function to call:"):].strip()
			open_bracket_index = function_call_line.find("[")
			close_bracket_index = function_call_line.find("]")
			function_call = function_call_line[open_bracket_index + 1:close_bracket_index].strip()
			open_parenthesis_index = function_call.find("(")
			close_parenthesis_index = function_call.find(")")

			function_name = function_call[:open_parenthesis_index].strip()
			arguments = function_call[open_parenthesis_index + 1:close_parenthesis_index].split(",")

			if function_name not in self.function_mapping.keys():
				function_name = None

		print(f"Function name: {function_name}, Arguments: {arguments}, Answer: {answer}")
		
		return {
			"function_name": function_name,
			"arguments": arguments,
			"answer": answer
		}

	@staticmethod
	def parse_final_answer(execution_output: str) -> str:
		# Extract the final answer from the execution output
		final_answer_index = execution_output.lower().find("answer:")

		if final_answer_index == -1:
			return execution_output
		else:
			return execution_output[final_answer_index + len("Answer:"):].strip()

	@staticmethod
	def split_general_actions(general_response: str) -> tuple:
		# Get all the lines that have a [ ] in them
		action_titles = [line[line.find("]:") + 1:].strip() for line in general_response.split("\n") if re.search(r"\[.*\]", line)]
		goal = action_titles[0]
		action_titles = action_titles[1:]

		return goal, action_titles
	
	def concatenate_execution_prompt(self, action_title: str, goal: str, past_action_results: list[str]) -> str:
		prompt = f"{self.execution_base_prompt}\nTask: {action_title}\nExtra information:\n- GOAL = {goal}"
		for result in past_action_results:
			prompt += f"\n- {result}"

		return prompt
	
	def execute_function(self, function_call: dict, image_path: str) -> str:
		# Execute the function and return the result
		if function_call["function_name"] in self.function_mapping:
			if any(arg for arg in function_call["arguments"]):
				return self.function_mapping[function_call["function_name"]](image_path, *function_call["arguments"])
			else:
				return self.function_mapping[function_call["function_name"]](image_path)
		else:
			raise Exception(f"Function {function_call['function_name']} not found.")
