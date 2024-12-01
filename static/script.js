document.addEventListener('DOMContentLoaded', () => {
    // Select DOM elements
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const resetButton = document.getElementById('reset-button');
    const imageInput = document.getElementById('image-input');
    const uploadButton = document.getElementById('upload-button');
    const mainImage = document.getElementById('main-image');
    const imageViewer = document.querySelector('.image-viewer');
    const imageEditor = document.getElementById('image-editor');
    const imageCanvas = document.getElementById('image-canvas');
    const drawMaskButton = document.getElementById('draw-mask-button');
    const addPointButton = document.getElementById('add-point-button');
    const saveEditButton = document.getElementById('save-edit-button');
    const closeEditorButton = document.getElementById('close-editor-button');
    const feedbackModal = document.getElementById('feedback-modal');
    const feedbackConfirm = document.getElementById('feedback-confirm');
    const feedbackReject = document.getElementById('feedback-reject');
    const feedbackImagesContainer = document.getElementById('feedback-images');
    const imageTaskbar = document.querySelector('.image-taskbar');
    const imagePlaceholder = document.getElementById('image-placeholder');
    let uploadedImages = [];
    let currentImageIndex = -1;
    let selectedImages = [];
    let activeEditorTool = null;
    let ctx = null;
    let isDrawing = false;
    let drawPoints = [];
    let maskCoordinates = [];
    let feedbackData = null;

       // Initialize canvas context
        function initializeCanvas() {
            if (imageCanvas && imageCanvas.getContext) {
                ctx = imageCanvas.getContext('2d');
            } else {
                console.error('Canvas or context not supported');
            }
        }
    initializeCanvas();

    // Helper function to create chat bubbles
     function displayChatMessage(message, sender = 'user', imageUrl = null) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('chat-message'); // Add the general chat message class
            messageDiv.classList.add(sender);  // Add 'user' or 'llm' class
        
            const contentDiv = document.createElement('div');
            contentDiv.classList.add('chat-message-content');
            contentDiv.textContent = message;
            messageDiv.appendChild(contentDiv);
            if (imageUrl) {
                const img = document.createElement('img');
                img.src = imageUrl;
                img.classList.add('chat-message-image');
                img.onload = () => {
                    contentDiv.appendChild(img);
                    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to bottom
                 // Add click event to enlarge the image
                    img.addEventListener('click', () => {
                        const expandedImg = new Image();
                        expandedImg.src = imageUrl;
                        expandedImg.style.maxWidth = '90vw';
                        expandedImg.style.maxHeight = '90vh';
                         const imgContainer = document.createElement('div');
                         imgContainer.style.position = 'fixed';
                         imgContainer.style.top = '0';
                         imgContainer.style.left = '0';
                         imgContainer.style.width = '100%';
                         imgContainer.style.height = '100%';
                         imgContainer.style.backgroundColor = 'rgba(0,0,0,0.8)';
                         imgContainer.style.display = 'flex';
                         imgContainer.style.justifyContent = 'center';
                         imgContainer.style.alignItems = 'center';
                         imgContainer.style.zIndex = '1000'; // Ensure it's on top

                         imgContainer.appendChild(expandedImg);

                         document.body.appendChild(imgContainer);
                        // Close when clicked outside
                        imgContainer.addEventListener('click', (event) => {
                             if (event.target === imgContainer) {
                                 document.body.removeChild(imgContainer);
                            }
                        });
                    });
                 };
                img.onerror = () => {
                    console.error('Error loading image:', imageUrl);
                    contentDiv.appendChild(document.createTextNode('(Image failed to load)'));
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                };
            } else {
                chatHistory.appendChild(messageDiv);
                chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to bottom
            }
        }

        function addImageToTaskbar(imageUrl, borderClass) {
            const taskbarItem = document.createElement('div');
            taskbarItem.classList.add('taskbar-item', borderClass); // Add border class
    
            const img = document.createElement('img');
            img.src = imageUrl;
            img.alt = 'Action Image';
            taskbarItem.appendChild(img);
    
            // You might want to add a click handler to view the larger image here as well
    
            imageTaskbar.appendChild(taskbarItem);
        }

     // Function to read and add an image
        function addImage(file) {
             const reader = new FileReader();
                reader.onload = function(e) {
                const newImage = {
                    file: file,
                    dataUrl: e.target.result,
                    mask: null, // To store mask data later
                    points: [] // To store point data later
                };
                uploadedImages.push(newImage);
                currentImageIndex = uploadedImages.length - 1; // Set as the current image
                loadImageToViewer(currentImageIndex);
                updateImageTaskbar();
                // If it's the first image, center the chat
                 if (uploadedImages.length === 1) {
                document.querySelector('.left-panel').style.width = '40%'; // Adjust width as needed
                document.querySelector('.right-panel').style.display = 'flex'; // Ensure right panel is visible
                 }
            };
             reader.readAsDataURL(file);
        }
    // Function to handle image upload
    function handleImageUpload(files) {
        Array.from(files).forEach(addImage);
    }
    // Function to update the image taskbar
    function updateImageTaskbar() {
        imageTaskbar.innerHTML = ''; // Clear existing items
        uploadedImages.forEach((imgData, index) => {
            const taskbarItem = document.createElement('div');
            taskbarItem.classList.add('taskbar-item');
            if(index === currentImageIndex) taskbarItem.classList.add('selected');
            const img = document.createElement('img');
            img.src = imgData.dataUrl;
            img.alt = `Image ${index + 1}`;
            taskbarItem.appendChild(img);

            // Add close button
            const closeButton = document.createElement('button');
            closeButton.innerHTML = '×';
            closeButton.classList.add('close-button');
            closeButton.addEventListener('click', (event) => {
                event.stopPropagation(); // Prevent taskbar item click
                removeImage(index);
            });
            taskbarItem.appendChild(closeButton);

            taskbarItem.addEventListener('click', () => {
                loadImageToViewer(index);
                const imageData = uploadedImages[index];
                const isSelected = selectedImages.includes(imageData);

                if (isSelected) {
                    selectedImages = selectedImages.filter(img => img !== imageData);
                    taskbarItem.classList.remove('selected-to-send');
                } else {
                    selectedImages.push(imageData);
                    taskbarItem.classList.add('selected-to-send');
                }
                updateSelectedImagesPreview();
            });
            imageTaskbar.appendChild(taskbarItem);
        });
    }

    function updateSelectedImagesPreview() {
        const previewContainer = document.getElementById('selected-images-preview');
        previewContainer.innerHTML = ''; // Clear previous previews
    
        selectedImages.forEach((imageData) => {
            const img = document.createElement('img');
            img.src = imageData.dataUrl;
            img.alt = 'Selected Image';
            img.classList.add('preview-thumbnail'); // Add class for styling/animation
            previewContainer.appendChild(img);
    
            // Simple fade-in animation
            img.style.opacity = 0;
            setTimeout(() => {
                img.style.opacity = 1;
            }, 100);
        });
    }

    // Function to load an image into the main viewer
    function loadImageToViewer(index) {
        if (index >= 0 && index < uploadedImages.length) {
            currentImageIndex = index;
            mainImage.src = uploadedImages[index].dataUrl;
            mainImage.classList.remove('hidden'); // Ensure it's visible
            updateImageTaskbar();
            // Update the selected state in the taskbar
             Array.from(imageTaskbar.children).forEach((item, i) => {
                if (i === index) {
                    item.classList.add('selected');
                } else {
                    item.classList.remove('selected');
                }
            });
             // Redraw mask and points if they exist
            redrawMaskAndPoints();
            mainImage.style.display = 'block'; // Show image when loaded
            imagePlaceholder.style.display = 'none'; // Hide placeholder
            mainImage.classList.add('loading');
                mainImage.onload = () => {
                mainImage.classList.remove('loading');
                    imageCanvas.width = mainImage.width;
                    imageCanvas.height = mainImage.height;
                drawImageScaled();
                redrawMaskAndPoints();
            };
        }
    }
    function removeImage(index) {
        if (index >= 0 && index < uploadedImages.length) {
            uploadedImages.splice(index, 1);
            if (uploadedImages.length === 0) {
                currentImageIndex = -1;
                mainImage.src = ''; // Clear main viewer
                 mainImage.classList.add('hidden');
                 document.querySelector('.left-panel').style.width = '100%'; // Occupy full width
                document.querySelector('.right-panel').style.display = 'none'; // Hide right panel
                mainImage.style.display = 'none'; // Hide image
                imagePlaceholder.style.display = 'block';
            } else {
                currentImageIndex = Math.max(0, index - 1); // Adjust current index
                loadImageToViewer(currentImageIndex);
            }
            updateImageTaskbar();
        }
    }
    // Function to start drawing on canvas
        function startDrawing(event) {
            isDrawing = true;
            const rect = imageCanvas.getBoundingClientRect();
             const scaleX = imageCanvas.width / mainImage.width;
            const scaleY = imageCanvas.height / mainImage.height;
            const x = (event.clientX - rect.left)/scaleX;
            const y = (event.clientY - rect.top)/scaleY;

            if (activeEditorTool === 'mask') {
                maskCoordinates = [];
                maskCoordinates.push({ x, y }); // Start a new mask path
            } else if (activeEditorTool === 'point') {
                drawPoints.push({ x, y });
                redrawMaskAndPoints(); // Immediately draw the point
            }
        }

        // Function to draw on canvas during mouse move
        function draw(event) {
            if (!isDrawing || !ctx) return;
            const rect = imageCanvas.getBoundingClientRect();
             const scaleX = imageCanvas.width / mainImage.width;
            const scaleY = imageCanvas.height / mainImage.height;
            const x = (event.clientX - rect.left)/scaleX;
            const y = (event.clientY - rect.top)/scaleY;

             if (activeEditorTool === 'mask') {
                maskCoordinates.push({ x, y });
                ctx.lineTo(x, y);
                ctx.stroke();
            }
        }

        // Function to stop drawing on canvas
        function stopDrawing() {
            isDrawing = false;
            if (activeEditorTool === 'mask') {
                ctx.closePath(); // Close the mask path
                 uploadedImages[currentImageIndex].mask = maskCoordinates; // Store the coordinates
            }
        }
         function redrawMaskAndPoints() {
            if (!ctx) return;

            ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height); // Clear the canvas
            drawImageScaled(); // Redraw the image first
            ctx.strokeStyle = 'red'; // Set color for mask and points
            ctx.lineWidth = 2;

            // Redraw mask if it exists
            const mask = uploadedImages[currentImageIndex].mask;
            if (mask) {
                ctx.beginPath();
                mask.forEach((point, index) => {
                    if (index === 0) {
                        ctx.moveTo(point.x, point.y);
                    } else {
                        ctx.lineTo(point.x, point.y);
                    }
                });
                ctx.closePath();
                ctx.stroke();
            }

            // Redraw points if they exist
            drawPoints = uploadedImages[currentImageIndex].points;
            drawPoints.forEach(point => {
                ctx.beginPath();
                ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI); // Draw a small circle for the point
                ctx.fill(); // Fill the circle
            });
        }
        function drawImageScaled() {
        if (!ctx || !mainImage.src || currentImageIndex === -1 ) return;

            const hRatio = imageCanvas.width / mainImage.width;
            const vRatio = imageCanvas.height / mainImage.height    ;
            const ratio  = Math.min ( hRatio, vRatio );
            const centerShift_x = ( imageCanvas.width - mainImage.width*ratio ) / 2;
            const centerShift_y = ( imageCanvas.height - mainImage.height*ratio ) / 2;
                ctx.clearRect(0,0,imageCanvas.width, imageCanvas.height);
                ctx.drawImage(mainImage, 0,0, mainImage.width, mainImage.height,
                               centerShift_x, centerShift_y, mainImage.width*ratio, mainImage.height*ratio);
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        const images = imageInput.files;
    
        if (!message && images.length === 0) {
            alert('Please enter a message or upload an image.');
            return;
        }
    
        const formData = new FormData();
        formData.append('message', message);
        selectedImages.forEach((imgData) => {
            formData.append('images', imgData.file); // Send selected image files
        });

        displayChatMessage(message, 'user');
    
        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Failed to process message.');

            const reader = response.body.getReader();
            let decoder = new TextDecoder("utf-8");
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                let chunk = decoder.decode(value);
                // Split the chunk into individual JSON objects (messages)
                let messages = chunk.split('\n').filter(msg => msg.trim() !== ""); 
                for (let message of messages) {
                    try {
                        let data = JSON.parse(message);
                        if (data.sender === 'bot') {
                            displayChatMessage(data.message, data.sender, data.imageUrl);
                        } else {
                            displayAction(data);
                            addImageToTaskbar(data.imageUrl, 'action-border'); // Add image to taskbar
                        }
                    } catch (e) {
                    console.error("Could not parse JSON:", message, e); // Handle possible parsing errors
                  }
                }
            }

        } catch (error) {
            console.error('Error sending message:', error);
        }
    
        userInput.value = ''; // Clear input field
        imageInput.value = ''; // Clear image input
    }
    
    // Update chat history in the UI
    function updateChatHistory(history) {
        chatHistory.innerHTML = ''; // Clear existing chat history
        history.forEach((entry) => {
            displayChatMessage(entry.message, entry.sender);
        });
    }

    function displayAction(actionData) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('action-message');  // Style for action updates
        messageDiv.innerHTML = `<strong>${actionData.title}</strong>: ${actionData.result}`; // Show task title & result

        if (actionData.imageUrl) {
            const img = document.createElement('img');
            img.src = actionData.imageUrl;
            img.classList.add('action-image');
            img.alt = actionData.title;
            // ... (Add click event for enlarged view as before if needed)
            messageDiv.appendChild(img);
        }

        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight; 
    }
    

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
    mainImage.style.display = 'none'; // Initially hide the main image
    imagePlaceholder.style.display = 'block';


    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    sendButton.addEventListener('click', sendMessage);
    resetButton.addEventListener('click', resetConversation);
    feedbackConfirm.addEventListener('click', () => {
        feedbackModal.classList.add('hidden');
        alert('Confirmed!');
    });
    feedbackReject.addEventListener('click', () => {
        feedbackModal.classList.add('hidden');
        alert('Cancelled!');
    });
    uploadButton.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', () => handleImageUpload(imageInput.files));
    mainImage.addEventListener('click', () => {
        imageViewer.classList.toggle('fullscreen');
        imageEditor.classList.remove('hidden');
         imageCanvas.width = mainImage.width;
         imageCanvas.height = mainImage.height;

            drawImageScaled();

    });
    closeEditorButton.addEventListener('click', () => {
         imageViewer.classList.remove('fullscreen');
        imageEditor.classList.add('hidden');
    });
    drawMaskButton.addEventListener('click', () => {
        activeEditorTool = 'mask';
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
    });

    addPointButton.addEventListener('click', () => {
        activeEditorTool = 'point';
    });
    saveEditButton.addEventListener('click', () => {
        if (currentImageIndex !== -1) {
            uploadedImages[currentImageIndex].points = drawPoints;
            uploadedImages[currentImageIndex].mask = maskCoordinates;
            // Additional logic to send updated image data to the backend (if needed)
        }
    });

    imageCanvas.addEventListener('mousedown', startDrawing);
    imageCanvas.addEventListener('mousemove', draw);
    imageCanvas.addEventListener('mouseup', stopDrawing);
    imageCanvas.addEventListener('mouseout', stopDrawing);

});