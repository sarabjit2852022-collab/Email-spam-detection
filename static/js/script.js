/* =============================================================
   Email Spam Detection - Frontend Logic
   Handles: real-time validation, AJAX prediction requests,
   loading animation, dynamic result rendering, clear/reset,
   and example message chips.
   ============================================================= */

document.addEventListener("DOMContentLoaded", () => {
    // -----------------------------------------------------------
    // Element references
    // -----------------------------------------------------------
    const emailInput = document.getElementById("emailInput");
    const charCount = document.getElementById("charCount");
    const validationMsg = document.getElementById("validationMsg");
    const predictBtn = document.getElementById("predictBtn");
    const clearBtn = document.getElementById("clearBtn");

    const emptyState = document.getElementById("emptyState");
    const resultContent = document.getElementById("resultContent");
    const resultBadge = document.getElementById("resultBadge");
    const resultIcon = document.getElementById("resultIcon");
    const resultLabel = document.getElementById("resultLabel");
    const confidenceText = document.getElementById("confidenceText");
    const confidenceFill = document.getElementById("confidenceFill");
    const hamProb = document.getElementById("hamProb");
    const spamProb = document.getElementById("spamProb");
    const resultNote = document.getElementById("resultNote");

    const MAX_CHARS = 3000;
    const MIN_CHARS = 3;

    // Example messages for the quick-try chips
    const EXAMPLES = {
        spam: "CONGRATULATIONS! You have WON a $1000 Walmart Gift Card. Click here www.claim-prize-now.com to claim your reward before it expires. Reply YES now!",
        ham: "Hey, just checking if we're still meeting for lunch tomorrow at noon? Let me know if that time still works for you. Thanks!",
    };

    // -----------------------------------------------------------
    // Real-time character counter + validation
    // -----------------------------------------------------------
    function validateInput() {
        const text = emailInput.value;
        const length = text.trim().length;

        charCount.textContent = text.length;

        if (text.length === 0) {
            validationMsg.textContent = "";
            emailInput.classList.remove("invalid");
            return false;
        }

        if (length < MIN_CHARS) {
            validationMsg.textContent = "Please enter at least 3 characters.";
            emailInput.classList.add("invalid");
            return false;
        }

        if (text.length > MAX_CHARS) {
            validationMsg.textContent = `Message is too long (max ${MAX_CHARS} characters).`;
            emailInput.classList.add("invalid");
            return false;
        }

        validationMsg.textContent = "";
        emailInput.classList.remove("invalid");
        return true;
    }

    emailInput.addEventListener("input", validateInput);

    // -----------------------------------------------------------
    // Predict button click -> call Flask /predict API
    // -----------------------------------------------------------
    predictBtn.addEventListener("click", async () => {
        const text = emailInput.value.trim();

        if (!validateInput() || text.length === 0) {
            validationMsg.textContent = "Please enter a valid message before predicting.";
            emailInput.classList.add("invalid");
            emailInput.focus();
            return;
        }

        setLoading(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email_text: text }),
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                showError(data.error || "Something went wrong. Please try again.");
                return;
            }

            renderResult(data);
        } catch (err) {
            showError("Unable to connect to the server. Please check your connection.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    });

    // -----------------------------------------------------------
    // Clear button -> reset form and result panel
    // -----------------------------------------------------------
    clearBtn.addEventListener("click", () => {
        emailInput.value = "";
        charCount.textContent = "0";
        validationMsg.textContent = "";
        emailInput.classList.remove("invalid");
        resetResultPanel();
        emailInput.focus();
    });

    // -----------------------------------------------------------
    // Example chips -> fill textarea with sample message
    // -----------------------------------------------------------
    document.querySelectorAll(".chip").forEach((chip) => {
        chip.addEventListener("click", () => {
            const type = chip.getAttribute("data-type");
            emailInput.value = EXAMPLES[type];
            validateInput();
            emailInput.focus();
        });
    });

    // -----------------------------------------------------------
    // Helper: toggle loading animation on the Predict button
    // -----------------------------------------------------------
    function setLoading(isLoading) {
        if (isLoading) {
            predictBtn.classList.add("loading");
            predictBtn.disabled = true;
        } else {
            predictBtn.classList.remove("loading");
            predictBtn.disabled = false;
        }
    }

    // -----------------------------------------------------------
    // Helper: render prediction result dynamically
    // -----------------------------------------------------------
    function renderResult(data) {
        emptyState.classList.add("hidden");
        resultContent.classList.remove("hidden");

        const isSpam = data.is_spam;

        // Badge styling: red for spam, green for ham
        resultBadge.classList.remove("spam", "ham");
        resultBadge.classList.add(isSpam ? "spam" : "ham");
        resultIcon.className = isSpam
            ? "fa-solid fa-triangle-exclamation"
            : "fa-solid fa-circle-check";
        resultLabel.textContent = isSpam ? "SPAM DETECTED" : "HAM (SAFE)";

        // Confidence bar
        confidenceText.textContent = `${data.confidence}%`;
        confidenceFill.classList.remove("spam-fill", "ham-fill");
        confidenceFill.classList.add(isSpam ? "spam-fill" : "ham-fill");

        // Animate the width after a tiny delay so the CSS transition triggers
        confidenceFill.style.width = "0%";
        requestAnimationFrame(() => {
            setTimeout(() => {
                confidenceFill.style.width = `${data.confidence}%`;
            }, 50);
        });

        // Probability breakdown
        hamProb.textContent = `${data.ham_probability}%`;
        spamProb.textContent = `${data.spam_probability}%`;

        // Friendly note
        resultNote.textContent = isSpam
            ? "This message shows strong characteristics of spam content. Avoid clicking any links or sharing personal information."
            : "This message appears to be a legitimate, safe communication.";
    }

    // -----------------------------------------------------------
    // Helper: reset result panel back to empty state
    // -----------------------------------------------------------
    function resetResultPanel() {
        resultContent.classList.add("hidden");
        emptyState.classList.remove("hidden");
        confidenceFill.style.width = "0%";
    }

    // -----------------------------------------------------------
    // Helper: show inline error message
    // -----------------------------------------------------------
    function showError(message) {
        validationMsg.textContent = message;
        emailInput.classList.add("invalid");
    }

    // Allow Ctrl+Enter / Cmd+Enter to trigger prediction
    emailInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            predictBtn.click();
        }
    });
});
