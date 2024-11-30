import os
from datetime import datetime
import streamlit as st
from PIL import Image
from llm import LLM

# Load LLM setup
llm = LLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5,
    max_tokens=150,
    system_prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."
)

# Directory for storing uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chat history structure
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# Helper function for LLM response
def ask_llm(question):
    try:
        response = llm.ask(question)
        return response
    except Exception as e:
        return f"Error: {e}"

# Helper for image uploads
def handle_image_upload(uploaded_file):
    if uploaded_file:
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_images.append(filepath)
        return filepath
    return None

# Apply custom CSS for styling
def apply_styles():
    st.markdown(
        """
        <style>
            body {
                background-color: #f9f9f9;
                font-family: 'Arial', sans-serif;
                color: #333;
            }
            .chat-container {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.05);
                margin-top: 20px;
            }
            .user-message, .bot-message {
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 10px;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.05);
            }
            .user-message {
                background-color: #d8eefe;
                color: #333;
            }
            .bot-message {
                background-color: #f0f0f0;
                color: #333;
            }
            .stTextInput > div > div > input {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 10px;
                background-color: #ffffff;
                color: #333;
            }
            .stFileUploader div div input {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 5px;
                background-color: #ffffff;
                color: #333;
            }
            h2, h3 {
                color: #005bb5;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Chat UI
def display_chat():
    st.markdown("<h2 style='text-align: center;'>Chat AI</h2>", unsafe_allow_html=True)
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            if message["is_user"]:
                st.markdown(
                    f"""
                    <div class='user-message'>
                        <strong>You:</strong> {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if message["image_path"]:
                    st.image(message["image_path"], caption="Uploaded Image", use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div class='bot-message'>
                        <strong>Bot:</strong> {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# Sidebar UI
def display_sidebar():
    with st.sidebar:
        st.title("Uploaded Images")
        if st.session_state.uploaded_images:
            for img_path in st.session_state.uploaded_images:
                img = Image.open(img_path)
                st.image(img, use_container_width=True, caption=os.path.basename(img_path))
        else:
            st.write("No images uploaded yet.")

# Main App
def main():
    st.set_page_config(page_title="AI Chatbot", layout="wide", initial_sidebar_state="expanded")
    apply_styles()

    display_sidebar()
    display_chat()

    # Message Input
    with st.form("chat_form", clear_on_submit=True):
        st.text_input("Type your message here:", placeholder="Ask something...", key="user_message")
        uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        submitted = st.form_submit_button("Send")

    if submitted:
        message = st.session_state.get("user_message", "").strip()
        if not message and not uploaded_file:
            st.warning("Please enter a message or upload an image.")
        else:
            # Save user message
            image_path = handle_image_upload(uploaded_file)
            st.session_state.chat_history.append({"content": message, "is_user": True, "image_path": image_path})

            # Get bot response
            if message:
                bot_response = ask_llm(message)
                st.session_state.chat_history.append({"content": bot_response, "is_user": False, "image_path": None})

            st.rerun()

if __name__ == "__main__":
    main()
