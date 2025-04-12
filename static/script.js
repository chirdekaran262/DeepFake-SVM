document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("uploadForm");
    const fileUpload = document.getElementById("file-upload");
    const fileName = document.getElementById("file-name");
    const loading = document.getElementById("loading");
    const resultBox = document.getElementById("result-box");
    const resultText = document.querySelector(".result-text");
    const confidenceBar = document.querySelector(".confidence-bar");
    const confidenceLabel = document.querySelector(".confidence-label");
    const flashMessages = document.getElementById("flash-messages");
    const darkModeToggle = document.getElementById("dark-mode-toggle");
    const body = document.body;

    // Update file name when a file is selected
    fileUpload.addEventListener("change", function () {
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;
            // Remove any previous results
            resetResults();
        } else {
            fileName.textContent = "No file chosen";
        }
    });

    // Handle form submission
    uploadForm.addEventListener("submit", function (event) {
        event.preventDefault();

        try {
            // Validate file type
            if (fileUpload.files.length === 0) {
                showError("Please select an audio file.");
                return;
            }

            const file = fileUpload.files[0];
            const fileType = file.type;

            if (!fileType.includes("audio")) {
                showError("Please upload an audio file (MP3, WAV, etc.)");
                return;
            }

            // Prepare form data
            const formData = new FormData(uploadForm);

            // Show loading animation
            resetResults();
            loading.style.display = "flex";

            // Add timeout to fetch request to prevent hanging
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

            // Submit form data
            fetch("/", {
                method: "POST",
                body: formData,
                signal: controller.signal
            })
                .then(response => {
                    clearTimeout(timeoutId);
                    if (!response.ok) {
                        throw new Error("Server error: " + response.status);
                    }
                    return response.json();
                })
                .then(data => {
                    // Hide loading animation
                    loading.style.display = "none";

                    if (data.success) {
                        displayResult(data);
                    } else {
                        showError(data.error || "An error occurred while processing the file.");
                    }
                })
                .catch(error => {
                    loading.style.display = "none";

                    // Handle different types of errors
                    if (error.name === 'AbortError') {
                        showError("Request timed out. The server took too long to respond.");
                    } else if (error.message.includes("Failed to fetch")) {
                        showError("Connection error. Make sure the server is running properly.");
                    } else {
                        showError("An error occurred: " + error.message);
                    }

                    console.error("Error:", error);
                });
        } catch (e) {
            loading.style.display = "none";
            showError("An unexpected error occurred: " + e.message);
            console.error("Submission error:", e);
        }
    });

    // Toggle dark/light mode
    darkModeToggle.addEventListener("click", function () {
        body.classList.toggle("dark-mode");
        const icon = darkModeToggle.querySelector("i");
        if (body.classList.contains("dark-mode")) {
            icon.className = "fas fa-sun";
            localStorage.setItem("theme", "dark");
        } else {
            icon.className = "fas fa-moon";
            localStorage.setItem("theme", "light");
        }
    });

    // Apply saved theme on page load
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "dark") {
        body.classList.add("dark-mode");
        darkModeToggle.querySelector("i").className = "fas fa-sun";
    } else {
        darkModeToggle.querySelector("i").className = "fas fa-moon";
    }

    // Hamburger menu toggle
    const hamburger = document.querySelector(".hamburger");
    const navLinks = document.querySelector(".nav-links");

    hamburger.addEventListener("click", function () {
        navLinks.classList.toggle("active");
    });

    // Display the analysis result
    function displayResult(data) {
        resultBox.style.display = "block";

        // Remove any previous result classes
        resultBox.classList.remove("result-real", "result-fake");

        const prediction = data.prediction.toLowerCase();
        const confidence = data.confidence || 75; // Default confidence if not provided

        // Set appropriate class and text based on the prediction
        if (prediction.includes("real")) {
            resultBox.classList.add("result-real");
            resultText.innerHTML = `<strong>REAL</strong> - This audio appears to be authentic`;
        } else {
            resultBox.classList.add("result-fake");
            resultText.innerHTML = `<strong>FAKE</strong> - This audio appears to be synthesized`;
        }

        // Update confidence label
        confidenceLabel.textContent = `${confidence}% Confidence`;

        // Animate the confidence bar
        setTimeout(() => {
            confidenceBar.style.width = "0%";
            confidenceBar.style.transition = "none";

            setTimeout(() => {
                confidenceBar.style.transition = "width 1s ease-in-out";
                confidenceBar.style.width = confidence + "%";
            }, 50);
        }, 10);

        // Smooth scroll to result
        resultBox.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    // Show error message
    function showError(message) {
        flashMessages.style.display = "block";
        flashMessages.textContent = message;
        flashMessages.scrollIntoView({ behavior: "smooth", block: "center" });

        // Auto-hide error after 8 seconds
        setTimeout(() => {
            flashMessages.style.display = "none";
        }, 8000);
    }

    // Reset previous results
    function resetResults() {
        resultBox.style.display = "none";
        flashMessages.style.display = "none";
    }
});