# 🧠 CIFAR-10 Classification using `gemma3:4b` (ai.sooners.us)

---

## 🎯 Goal

Use an OpenAI-compatible API at [https://ai.sooners.us](https://ai.sooners.us) with the **gemma3:4b** model (a Vision-Language Model) to:

- Classify **100 images from CIFAR-10** (10 images from each of the 10 classes)
- Experiment with different **system prompts** to improve accuracy
- Plot and save a **confusion matrix** from the results

Each image is sent to the API (as Base64) in a chat-completion request, and the model’s label is parsed from the response.

---


## 1️⃣ Setup Steps

### 🧩 Environment Setup

Create a hidden environment file to store your API key and configuration:

**File path:**
`~/.soonerai.env`

**Contents:**
```bash
SOONERAI_API_KEY=your_key_here
SOONERAI_BASE_URL=https://ai.sooners.us
SOONERAI_MODEL=gemma3:4b
```
## 2️⃣ How to Run the Code

### 🚀 Execution Steps

Once your environment and dependencies are set up, run the main script to start classification.

**Command:**
```bash
python3 cifar10_classify.py
```
## 3️⃣ Analysis

### 🧠 System Prompts Tested

Experimented with different system prompts to evaluate how wording affects classification accuracy.

---

### 🟢 **Prompt 1 – Concise Label Only**

A short, direct instruction to make the model reply with only one class label.

**Prompt:**
```text
You are a precise image classifier for CIFAR-10.
Reply with exactly one label from:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
```
### 🔵 **Prompt 2 – Explain Then Label**

A reasoning-based instruction that asks the model to first describe the image briefly and then provide a final classification label.

**Prompt:**
```text
You are an image analyst.
Describe the image briefly, then on a new line output exactly one label from:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
The final line must contain only the label.
```

## 4️⃣ Results and Confusion Matrix

### 📈 Accuracy Evaluation

After classifying 100 images (10 from each of the 10 CIFAR-10 classes), the model’s performance was measured using different prompts.

| Prompt | Accuracy | Key Observation |
|---------|-----------|----------------|
| 🟢 Concise Label Only | 63% | Clean, one-word outputs; fast and consistent |
| 🔵 Explain Then Label | 56% | Slightly better accuracy; reasoning helps identify tricky images |


---

### 📊 Confusion Matrix

A **confusion matrix** was generated to visualize misclassifications across the 10 CIFAR-10 categories.

**Saved File:**

Each row of the matrix represents the **true class**, while each column represents the **predicted class**.  
Diagonal values indicate correct predictions; off-diagonal values indicate misclassifications.

---

### 🧩 Common Misclassifications

| True Class | Predicted Class | Explanation |
|-------------|----------------|--------------|
| automobile | truck | Both share similar shapes and backgrounds |
| bird | airplane | Sky backgrounds often confuse the model |
| cat | dog | Similar size, color, and pose in CIFAR-10 images |

---

### 💬 Summary
- The **Explain Then Label** prompt slightly improves overall performance by encouraging reasoning.  
- The **Concise Label** prompt ensures simple and consistent outputs that are easier to parse automatically.  
- Future improvement: combine brevity and reasoning in one hybrid prompt.

---

## 5️⃣ Security and Reproducibility

### 🔒 API Key Handling

- Your API key is stored securely in the `~/.soonerai.env` file.  
- It is **never hardcoded** in the code or uploaded to GitHub.  
- This ensures your key remains private and prevents unauthorized access.

---

### ⚙️ Reproducibility

To ensure the experiment can be replicated exactly:
- The script uses a **fixed random seed (1337)** for image sampling.  
- The **model temperature** is set to `0.0` for deterministic responses.  
- CIFAR-10 dataset automatically downloads from the official PyTorch source if not available locally.

---

### 🧾 .gitignore Configuration

Ensure your `.gitignore` file includes the following entries to protect sensitive files and reduce repository clutter:
```text
*.env
pycache/
venv/
data/
results/
.ipynb_checkpoints/
```
