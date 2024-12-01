import os
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, url_for, copy_current_request_context
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from controller import Controller  # Import your Controller class
import json
import matplotlib.pyplot as plt
import numpy as np
import io

import base64
# Load environment variables
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the Controller
controller = Controller()

# Chat history to store the conversation
chat_history = []
current_request = None

@app.route('/')
def home():
    """Serve the chatbot interface."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_message():
    global chat_history
    global current_request  # Use the global variable
    data = request.form
    files = request.files.getlist("images")
    
    current_request = request

    # Save uploaded images
    image_paths = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(file_path)
        image_paths.append(file_path)
    print(image_paths)

    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400
    
    chat_history.append({"sender": "user", "message": user_message})
    
    def generate_response(controller, user_message, image_path):
        controller.general_llm.clear_history()
        controller.execution_llm.clear_history()

        general_response = controller.general_llm.ask(controller.general_base_prompt_file + user_message)
        filtered_general_response = controller.filter_general_response(general_response)
        goal, action_titles = controller.split_general_actions(filtered_general_response)

        for i, action_title in enumerate(action_titles):
            try:
                execution_response = controller.execution_llm.ask(controller.concatenate_execution_prompt(action_title, goal, [controller.actions[j]["result"] for j in range(i)]))
                function_call_string = controller.parse_function_call(execution_response)

                if function_call_string["function_name"] == "segment_unique":
                    # Special handling for segment_unique function
                    result, segmented_images = controller.execute_function(function_call_string, image_path)
                    
                    feedback_data = {
                        "title": action_title,
                        "message": "Please select a segmentation mask:",
                        "images": [],  # Store encoded images
                        "original_image": image_path
                    }

                    print("here", feedback_data)
                    for i, img in enumerate(segmented_images):
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        feedback_data["images"].append(img_str)

                    print("next", feedback_data)

                    yield json.dumps({"feedback_required": True, "feedback_data": feedback_data}) + '\n'

                    # Wait for feedback from the client
                    def wait_for_feedback():
                        global current_request
                        feedback_response = current_request.get_json()
                        selected_mask_index = feedback_response.get("selected_mask_index")
                        print(selected_mask_image)
                        print("RESPONSE", feedback_response)

                        if selected_mask_index is not None:

                            selected_mask_image = segmented_images[selected_mask_index]
                            
                            final_segmented_image_path = image_path.replace(".jpg", "_unique_segmented.jpg").replace(".png", "_unique_segmented.png")

                            plt.imsave(final_segmented_image_path, np.array(selected_mask_image))
                            action_data = {
                                "title": action_title,
                                "image_url": url_for('uploaded_file', filename=os.path.basename(final_segmented_image_path)), # URL for the saved image
                                "result": f'Selected segmentation mask {selected_mask_index + 1} applied.'
                            }
                            controller.actions.append(action_data)
                            controller.num_actions += 1
                            yield json.dumps(action_data) + '\n'

                        else:
                            # Handle invalid feedback (e.g., timeout, incorrect format)
                            error_data = {
                                "title": action_title,
                                "error": "Invalid feedback received."
                            }
                            yield json.dumps(error_data) + '\n'
                    wait_for_feedback()
                else:
                    result, image_path_new = controller.execute_function(function_call_string, image_path)
                    print(image_path)
                    action_data = {
                        "title": action_title,
                        "function_name": function_call_string["function_name"],
                        "arguments": function_call_string["arguments"],
                        "answer": function_call_string["answer"],
                        "image_url": image_path_new, # include image URL
                        "result": result
                    }
                    controller.actions.append(action_data)
                    controller.num_actions += 1
                    
                    # Yield intermediate action data as JSON
                    print(image_path_new)
                    yield json.dumps(action_data) + '\n'
            except Exception as e:  # Catch specific exception for each action
                print(f"Error executing action {action_title}: {e}") # Log details!
                error_data = {
                    "title": action_title,
                    "error": str(e)
                }
                yield json.dumps(error_data) + '\n' # Send error info to the client
        
        final_answer = controller.execution_llm.ask(controller.concatenate_execution_prompt("Now with all the information you must answer the question in order to achieve the GOAL.", goal, [controller.actions[j]["result"] for j in range(controller.num_actions)]))
        final_answer_data = {"sender": "bot", "message": controller.parse_final_answer(final_answer), "imageUrl": image_path} # Final answer with image
        chat_history.append(final_answer_data)
        print(chat_history)
        print("AAAAAAAAA")
        print(final_answer_data)
        yield json.dumps(final_answer_data) + '\n'
        


    return Response(generate_response(controller, user_message, image_paths[0] if image_paths else None), mimetype='application/json')


@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset the chat history and uploaded images."""
    global chat_history
    chat_history = []
    
    # Clear uploaded files
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        os.remove(file_path)
    
    return jsonify({"message": "Conversation reset successfully."})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)
