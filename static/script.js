document.addEventListener("DOMContentLoaded", () => {
    // Chatbox functionality
    const chatWindow = document.getElementById("chat-window");
    const messagesDiv = document.getElementById("messages");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const printButton = document.getElementById("print-button");
    const interactiveContainer = document.querySelector(".interactive-container");


    // Send message
    sendButton.addEventListener("click", () => {
        const userMessage = userInput.value.trim();
        if (userMessage) {
            addMessage("You", userMessage);
            userInput.value = "";

            // Send the message to the Flask backend
            fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ user_input: userMessage }),
            })
                .then((response) => response.json())
                .then((data) => {
                    addMessage("MommaAI", data.response);
                })
                .catch((error) => console.error("Error communicating with server:", error));
        }
    });

    // Add message to chat
    function addMessage(sender, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message");
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        messagesDiv.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        if (interactiveContainer) {
            interactiveContainer.addEventListener("click", (e) => {
                if (e.target.classList.contains("interactive-choice")) {
                    const choice = e.target.innerText;
                    addMessage("You", choice);

                    // Fetch the story content from the backend
                fetch("/story", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ choice: choice }),
                })
                    .then((storyResponse) => response.json())
                    .then((data) => {
                        addMessage("MommaAI", storyResponse);
                    })
                    
                    // Simulate AI response based on choice
                    let storyResponse;
                    switch (choice) {
                        case "Explore the cave":
                            storyResponse = "You step into the dark, damp cave, and the echo of your footsteps awakens something deep within. A mysterious growl grows louder. What will you do next?";
                            break;
                        case "Climb the mountain":
                            storyResponse = "As you ascend the rugged mountain trail, the air gets thinner, and a breathtaking view unfolds. But wait! A sudden landslide blocks your path. What’s your move?";
                            break;
                        case "Sail across the lake":
                            storyResponse = "You set sail across the serene lake, but dark clouds quickly gather above. A storm brews, and your boat starts to rock. How will you survive?";
                            break;
                        default:
                            storyResponse = "Interesting choice! What happens next is up to your imagination...";
                    }

                    addMessage("MommaAI", storyResponse);
                }
            });
    }

    // Print chat
    printButton.addEventListener("click", () => {
        const printContent = messagesDiv.innerHTML;
        const printWindow = window.open("", "", "width=600,height=400");
        printWindow.document.write(`<html><body>${printContent}</body></html>`);
        printWindow.document.close();
        printWindow.print();
    });

    }

    // Photo gallery
    const photoGallery = document.getElementById("photo-gallery");
    const photosFolder = "/static/photos";

    if (photoGallery) {
        fetch(photosFolder)
            .then((response) => response.json())
            .then((photos) => {
                photos.forEach((photo) => {
                    const img = document.createElement("img");
                    img.src = `/static/photos/${photo}`;
                    img.alt = "Photo";
                    img.classList.add("gallery-photo");
                    photoGallery.appendChild(img);
                });
            })
            .catch((error) => console.error("Error loading photos:", error));
    }

    // Photo gallery hover effect
    if (photoGallery) {
        photoGallery.addEventListener("mouseover", (e) => {
            if (e.target.tagName === "IMG") {
                e.target.style.border = "2px solid #007bff";
            }
        });
        photoGallery.addEventListener("mouseout", (e) => {
            if (e.target.tagName === "IMG") {
                e.target.style.border = "none";
            }
        });
    }

    // Audio player
    const audioPlayer = document.getElementById("audio-player");
    const audiosFolder = "/static/audios";

    if (audioPlayer) {
        fetch(audiosFolder)
            .then((response) => response.json())
            .then((audios) => {
                audios.forEach((audio) => {
                    const audioElement = document.createElement("audio");
                    audioElement.controls = true;
                    audioElement.src = `/static/audios/${audio}`;
                    audioPlayer.appendChild(audioElement);
                });
            })
            .catch((error) => console.error("Error loading audios:", error));
    }
});
