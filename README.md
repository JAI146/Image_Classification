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
