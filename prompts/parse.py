# Example function definition for demonstration
def count_people(queue_description):
    # Simulating a function call, returning a result
    return f"Counted people in {queue_description}"

def parse_and_execute(message):
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
    paragraphs = message.strip().split("\n\n")
    
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
    
    # Map the function name to the actual callable function
    function_mapping = {
        "count_people": count_people
    }
    
    # Execute the function dynamically
    if function_name in function_mapping:
        function_result = function_mapping[function_name](argument)
    else:
        raise ValueError(f"Function '{function_name}' is not defined.")
    
    return {
        "action_description": action_description,
        "function_result": function_result
    }

# Example usage
message = """
Next Action: I will begin by identifying the number of people in the queue. This will provide the basis for further calculations regarding processing times.

Tasks: [1]: Identify the number of people in the queue - Status: Completed - Output: [count_people('queue')] [2]: Estimate the average processing time per person in the queue - Status: Pending - Output: None [3]: Calculate the total time needed for the queue to end based on the number of people and processing time - Status: Pending - Output: None [Answer]: Provide an estimate of how much time it will take for the queue to end based on the calculations - Status: Pending - Output: None

Function to Call: [count_people('queue')]
"""

parsed_output = parse_and_execute(message)
print(parsed_output)
