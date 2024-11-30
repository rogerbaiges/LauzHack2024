document.addEventListener("DOMContentLoaded", () => {
    const chatHistory = document.getElementById("chat-history");
    const textInput = document.getElementById("text-input");
    const sendBtn = document.getElementById("send-btn");
    const imageInput = document.getElementById("image-input");
    const imagePreview = document.getElementById("image-preview");
    const resetBtn = document.getElementById("reset-btn");
    const popupContainer = document.getElementById("popup-container");
    const popupMessage = document.getElementById("popup-message");
    const popupImage = document.getElementById("popup-image");
    const popupConfirm = document.getElementById("popup-confirm");
    const popupCancel = document.getElementById("popup-cancel");

    let chatMessages = [];
    let uploadedImages = [];

    const renderChat = () => {
        chatHistory.innerHTML = chatMessages
            .map(msg => `<div class="message ${msg.type}">${msg.content}</div>`) 
            .join("");
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };

    const renderImagePreview = () => {
        imagePreview.innerHTML = uploadedImages
            .map((img, index) => `<img src="${img}" alt="Image ${index + 1}">`)
            .join("");
    };

    const sendMessage = () => {
        const text = textInput.value.trim();
        if (text) {
            chatMessages.push({ type: "user-message", content: text });
            textInput.value = "";
            renderChat();

            // Simulate bot response
            setTimeout(() => {
                chatMessages.push({ type: "bot-message", content: "Processing your message..." });
                renderChat();
            }, 500);
        }
    };

    const handleImageUpload = event => {
        const files = event.target.files;
        for (const file of files) {
            const reader = new FileReader();
            reader.onload = () => {
                uploadedImages.push(reader.result);
                renderImagePreview();
            };
            reader.readAsDataURL(file);
        }
    };

    const resetChat = () => {
        chatMessages = [];
        uploadedImages = [];
        renderChat();
        renderImagePreview();
    };

    const showPopup = (message, imageUrl) => {
        popupMessage.textContent = message;
        if (imageUrl) {
            popupImage.src = imageUrl;
            popupImage.classList.remove("hidden");
        } else {
            popupImage.classList.add("hidden");
        }
        popupContainer.classList.remove("hidden");
    };

    const hidePopup = () => {
        popupContainer.classList.add("hidden");
    };

    sendBtn.addEventListener("click", sendMessage);
    imageInput.addEventListener("change", handleImageUpload);
    resetBtn.addEventListener("click", resetChat);
    popupConfirm.addEventListener("click", () => {
        hidePopup();
        chatMessages.push({ type: "bot-message", content: "You confirmed the action." });
        renderChat();
    });
    popupCancel.addEventListener("click", () => {
        hidePopup();
        chatMessages.push({ type: "bot-message", content: "You canceled the action." });
        renderChat();
    });

    // Example backend-triggered popup
    setTimeout(() => {
        showPopup("Do you want to proceed with this action?", null);
    }, 3000);
});
