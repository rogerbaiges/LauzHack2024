import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from llm import LLM

# Initialize the Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the LLM
llm = LLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5,
    max_tokens=100
)

# Chat history to store the conversation
chat_history = []

@app.route('/')
def home():
    """Serve the chatbot interface."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_message():
    """Handle text and image input from the user."""
    global chat_history
    data = request.form
    files = request.files.getlist("images")

    # Process text input
    user_message = data.get("message", "").strip()
    if user_message:
        chat_history.append({"sender": "user", "message": user_message})
        # Get AI response
        response = llm.ask(user_message)
        chat_history.append({"sender": "bot", "message": response})

    # Process image input
    image_urls = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image_urls.append(f"/uploads/{filename}")

    # Return updated chat history and image URLs
    return jsonify({
        "chat_history": chat_history,
        "image_urls": image_urls
    })

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset the chat history and uploaded images."""
    global chat_history
    chat_history = []
    
    # Clear uploaded files
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        os.remove(file_path)
    
    # Clear LLM history
    llm.clear_history()
    
    return jsonify({"message": "Conversation reset successfully."})

@app.route('/trigger-popup', methods=['POST'])
def trigger_popup():
    """Backend-triggered popup functionality."""
    data = request.json
    question = data.get("question", "Default popup question")
    image_url = data.get("image_url", None)
    return jsonify({"question": question, "image_url": image_url})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
