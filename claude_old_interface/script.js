// Select DOM elements
const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const resetButton = document.getElementById('reset-button');
const imageInput = document.getElementById('image-input');
const imagePreview = document.getElementById('image-preview');
const popupModal = document.getElementById('popup-modal');
const popupQuestion = document.getElementById('popup-question');
const popupImagePreview = document.getElementById('popup-image-preview');
const confirmButton = document.getElementById('confirm-button');
const cancelButton = document.getElementById('cancel-button');

// Helper function to create chat bubbles
function createChatBubble(sender, message) {
    const bubble = document.createElement('div');
    bubble.classList.add('chat-bubble', sender);
    bubble.textContent = message;
    chatHistory.appendChild(bubble);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the bottom
}

// Helper function to preview uploaded images
function previewImages(files) {
    imagePreview.innerHTML = ''; // Clear previous previews
    Array.from(files).forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.alt = `Image ${index + 1}`;
            imagePreview.appendChild(img);
        };
        reader.readAsDataURL(file);
    });
}

// Send user input to the backend
async function sendMessage() {
    const message = userInput.value.trim();
    const images = imageInput.files;

    if (!message && images.length === 0) {
        alert('Please enter a message or upload an image.');
        return;
    }

    const formData = new FormData();
    formData.append('message', message);
    Array.from(images).forEach((file) => {
        formData.append('images', file);
    });

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) throw new Error('Failed to process message.');

        const data = await response.json();
        updateChatHistory(data.chat_history);
        updateImagePreview(data.image_urls);
    } catch (error) {
        console.error('Error sending message:', error);
    }

    userInput.value = ''; // Clear input field
    imageInput.value = ''; // Clear image input
    imagePreview.innerHTML = ''; // Clear image preview
}

// Update chat history in the UI
function updateChatHistory(history) {
    chatHistory.innerHTML = ''; // Clear existing chat history
    history.forEach((entry) => {
        createChatBubble(entry.sender, entry.message);
    });
}

// Update image previews for uploaded images
function updateImagePreview(urls) {
    imagePreview.innerHTML = ''; // Clear previous previews
    urls.forEach((url) => {
        const img = document.createElement('img');
        img.src = url;
        img.alt = 'Uploaded Image';
        imagePreview.appendChild(img);
    });
}

// Reset the chat and uploaded images
async function resetConversation() {
    try {
        const response = await fetch('/reset', {
            method: 'POST',
        });

        if (!response.ok) throw new Error('Failed to reset conversation.');

        const data = await response.json();
        console.log(data.message);

        chatHistory.innerHTML = ''; // Clear chat history
        imagePreview.innerHTML = ''; // Clear image previews
    } catch (error) {
        console.error('Error resetting conversation:', error);
    }
}

// Handle backend-triggered popup
async function handlePopup() {
    try {
        const response = await fetch('/trigger-popup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({}),
        });

        if (!response.ok) throw new Error('Failed to trigger popup.');

        const data = await response.json();
        popupQuestion.textContent = data.question;
        popupModal.classList.remove('hidden');

        if (data.image_url) {
            const img = document.createElement('img');
            img.src = data.image_url;
            img.alt = 'Popup Image';
            popupImagePreview.innerHTML = '';
            popupImagePreview.appendChild(img);
        } else {
            popupImagePreview.innerHTML = ''; // Clear if no image
        }
    } catch (error) {
        console.error('Error triggering popup:', error);
    }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
resetButton.addEventListener('click', resetConversation);
imageInput.addEventListener('change', () => previewImages(imageInput.files));
confirmButton.addEventListener('click', () => {
    popupModal.classList.add('hidden');
    alert('Confirmed!');
});
cancelButton.addEventListener('click', () => {
    popupModal.classList.add('hidden');
    alert('Cancelled!');
});
