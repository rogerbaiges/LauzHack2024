from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from llm import LLM
from datetime import datetime

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize LLM
llm = LLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5,
    max_tokens=100
)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Route for serving static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """Save uploaded file and return the filename"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to filename to prevent duplicates
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        message = request.form.get('message', '')
        files = request.files.getlist('images')
        
        # Save uploaded images
        image_paths = []
        for file in files:
            filepath = save_uploaded_file(file)
            if filepath:
                image_paths.append(filepath)
        
        # Prepare prompt for LLM
        prompt = message
        if image_paths:
            prompt += f"\n[User uploaded {len(image_paths)} images]"
        
        # Get response from LLM
        response = llm.ask(prompt)
        
        # Check if we need to show a popup
        # This is a simple example - you might want to implement more sophisticated logic
        should_show_popup = "confirm" in response.lower() or "verify" in response.lower()
        
        result = {
            "response": response,
            "popup": None
        }
        
        if should_show_popup:
            result["popup"] = {
                "title": "Confirmation Required",
                "message": "Would you like to proceed with this action?",
                "image": image_paths[0] if image_paths else None
            }
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Error processing message: {str(e)}")
        return jsonify({
            "error": "An error occurred while processing your message.",
            "response": "I apologize, but I encountered an error while processing your message. Please try again."
        }), 500

@app.route('/popup_response', methods=['POST'])
def popup_response():
    try:
        user_response = request.json.get('response', False)
        
        # Process the user's response
        response = llm.ask(f"User {'confirmed' if user_response else 'declined'} the action.")
        
        return jsonify({"response": response})
    
    except Exception as e:
        app.logger.error(f"Error processing popup response: {str(e)}")
        return jsonify({
            "error": "An error occurred while processing your response.",
            "response": "I apologize, but I encountered an error while processing your response."
        }), 500

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    try:
        # Clear LLM history
        llm.clear_history()
        
        # Clean up uploaded files
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                app.logger.error(f"Error deleting file {file_path}: {str(e)}")
        
        return jsonify({"status": "success"})
    
    except Exception as e:
        app.logger.error(f"Error resetting chat: {str(e)}")
        return jsonify({
            "error": "An error occurred while resetting the chat.",
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)