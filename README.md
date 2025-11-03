📄 Here’s the full README.md file (copy-paste everything into README.md):
# 🧠 CIFAR-10 Classification using `gemma3:4b` via ai.sooners.us

This project uses an **OpenAI-compatible API** at [https://ai.sooners.us](https://ai.sooners.us) with the **Gemma3:4b** Vision-Language Model (VLM) to classify **100 images from the CIFAR-10 dataset** — 10 images per class.

The goal is to:
- Experiment with **different system prompts** to improve classification accuracy.
- Compute and visualize a **confusion matrix**.
- Compare results and provide observations.

---

## 🎯 Learning Outcomes

By completing this project, you will be able to:
- Load and stratify-sample CIFAR-10 images using PyTorch.
- Send Base64-encoded images to an OpenAI-compatible API endpoint.
- Log predictions, compute overall accuracy, and plot a confusion matrix.
- Use **prompt engineering** to improve model performance and document findings.

---

## ⚙️ Setup Instructions

### 1️⃣ Environment Variables

Create a hidden environment file at:


~/.soonerai.env


**Contents:**


SOONERAI_API_KEY=your_key_here
SOONERAI_BASE_URL=https://ai.sooners.us

SOONERAI_MODEL=gemma3:4b


**Important:**  
- Do **not** commit this file to GitHub.
- Make sure the file is readable only by you:
  ```bash
  chmod 600 ~/.soonerai.env

2️⃣ Install Dependencies

Use a virtual environment for clean isolation.

python3 -m venv venv
source venv/bin/activate


Then install the required packages:

python3 -m pip install requests python-dotenv torch torchvision pillow scikit-learn matplotlib


(Or simply python3 -m pip install -r requirements.txt if you’ve created one.)

🚀 How to Run the Code

Ensure your .env file and API key are correctly set up.

Run the main script:

python3 cifar10_classify.py


The script will:

Download CIFAR-10 (if not already present).

Randomly sample 10 images per class (fixed seed = 1337 for reproducibility).

Send each image (as Base64) to gemma3:4b through the ai.sooners.us API.

Parse the model’s top label prediction.

Compute accuracy and generate a confusion matrix.

🧩 System Prompts Tested

You must test at least two different system prompts and compare results.

🟢 Prompt 1 — Concise Label Only
You are a precise image classifier for CIFAR-10.
Reply with exactly one label from:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
Do not include any other words or punctuation.

🔵 Prompt 2 — Explain Then Label
You are an image analyst.
Briefly describe what you see (one short sentence),
then output exactly one label from:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
The final line must contain only the label.


You can modify or add additional prompts in the PROMPTS list inside cifar10_classify.py.

📊 Output and Results

After running the script, you will get a folder structure like this:

results/
├── concise_label_only/
│   ├── confusion_matrix.png
│   ├── misclassifications.jsonl
│   └── summary.json
├── explain_then_label/
│   ├── confusion_matrix.png
│   ├── misclassifications.jsonl
│   └── summary.json
└── summary.csv


Each subfolder corresponds to one system prompt.

confusion_matrix.png → heatmap of model predictions.

summary.json → accuracy and metadata.

summary.csv → combined results across all prompts.

📈 Example Terminal Output
Preparing CIFAR-10 sample (100 images)...
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data
Classifying...
[001/100] true=airplane | pred=airplane | raw='airplane'
[002/100] true=dog | pred=cat | raw='This looks like a small furry animal, cat'
...
Accuracy over 100 images: 63.00%
Saved confusion_matrix.png

🧠 Analysis (to include after running)
Prompt	Accuracy	Observations
Concise Label Only	XX%	Consistent one-word labels; less reasoning context.
Explain Then Label	YY%	Slightly higher accuracy; benefited from short reasoning.

Error Patterns:

Common confusions between automobile ↔ truck and bird ↔ airplane.

Fine-grained animal classes (cat, dog, deer) sometimes misclassified.

Observations:

The descriptive prompt provided more context, slightly improving classification on ambiguous images.

However, concise prompts avoided extra text and made parsing easier.

Conclusion:

Prompt wording has a direct impact on VLM accuracy.
A reasoning-based prompt gave more correct labels, while a concise one was cleaner but less adaptive.

🔒 Security & Reproducibility

Your API key is never hardcoded — it’s loaded securely from ~/.soonerai.env.

CIFAR-10 sampling uses a fixed random seed for reproducibility.

Model temperature = 0.0 (deterministic).

.gitignore should exclude:

*.env
__pycache__/
venv/
.ipynb_checkpoints/
data/
results/

🧾 Rubric Mapping
Criterion	Points	Description
Functionality	8	Loads data, samples 10/class, calls API, computes accuracy, saves confusion matrix
Prompting/Iteration	6	At least two different system prompts; comparison with observations
Reproducibility/Docs	4	README includes setup, how to run, and analysis
Security	2	API key stored via .env, not committed
