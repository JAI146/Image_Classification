cat > README.md <<'README'
# 🧠 CIFAR-10 Image Classification using `gemma3:4b` (ai.sooners.us)

This project performs **image classification** on the CIFAR-10 dataset using the **Gemma3:4b** Vision-Language Model (VLM) via the OpenAI-compatible API hosted at **https://ai.sooners.us**.  

It evaluates how **different system prompts** affect model performance by classifying 100 images (10 per class) and comparing the results.

---

## 🎯 Project Goals

- Load and stratify-sample 100 images (10 from each CIFAR-10 class).  
- Send each image (Base64-encoded) to the API at `ai.sooners.us` using the `gemma3:4b` model.  
- Test **multiple system prompts** to study prompt-engineering effects on accuracy.  
- Compute overall accuracy and plot a **confusion matrix**.  
- Compare and discuss observations between prompts.

---

## 📂 Files Included

| File | Description |
|------|--------------|
| `cifar10_classify.py` | Main script that downloads CIFAR-10, samples 100 images, sends them to the API, parses predictions, and saves metrics. |
| `requirements.txt` | Python dependencies. |
| `README.md` | Setup instructions, run guide, and final analysis. |
| `.gitignore` | Keeps secrets (`*.env`, `venv/`, etc.) out of Git. |
| `~/.soonerai.env` | **Local only (not committed).** Stores your API key and base URL. |
| `results/` | Auto-generated after running the script. Contains confusion matrices, summaries, and misclassifications. |

---

## ⚙️ Environment Setup

### 1️⃣ Create your `.soonerai.env` file

In your terminal:
```bash
echo "SOONERAI_API_KEY=your_key_here
SOONERAI_BASE_URL=https://ai.sooners.us
SOONERAI_MODEL=gemma3:4b" > ~/.soonerai.env
chmod 600 ~/.soonerai.env
