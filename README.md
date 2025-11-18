# 🧩 Custom LLM Fine-Tuning with LoRA (Phi-2)

This repository demonstrates how to fine-tune the **Microsoft Phi-2** model using **Low-Rank Adaptation (LoRA)** on a custom dataset of 500 Q&A prompts.  
It also includes a simple inference module and a web app (Flask UI) to interact with the fine-tuned model.

## Table of Contents
	1.	Overview
	2.	Repository layout
	3.	How LoRA works
	4.	Setup (virtualenv / dependencies)
	5.	Training (train.py)
	6.	Dataset format
	7.	Inference (infer.py)
	8.	Web app (app.py)
	9.	Running in PyCharm
	10.	Makefile & helper scripts
	11.	Docker (CPU-only / Mac-safe)
	12.	Troubleshooting & FAQ

### 1 Overview

This project fine-tunes microsoft/phi-2 using LoRA adapters on a 500-sample, role-aware Q&A dataset (data/train_qa_500.jsonl). It demonstrates:
	•	Efficient fine-tuning using PEFT / LoRA.
	•	Saving adapters separately so the base model remains intact.
	•	Role-aware inference (TPM, PM, Software Engineer, Data Scientist, ML Engineer).
	•	Local serving with Flask and optional Docker deployment.

### 2 Repository layout

```
custom-llm-with-lora/
│
├── data/
│   └── train_500.jsonl
├── finetuned_phi2_lora_500
├── train.py
├── infer.py
├── infer_phi2_with_roles.py
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── scripts/
    ├── train.sh
    └── serve.sh
```
### 3 How LoRA Works

#### 🔹 Why Not Fine-Tune the Whole Model?
Large language models like **Phi-2** have **billions of parameters**. Fine‑tuning all weights:
- requires massive GPU resources,
- is slow,
- increases the risk of catastrophic forgetting,
- and produces a huge output model.

#### 🔹 Enter LoRA (Low-Rank Adaptation)
LoRA is a technique that freezes **all original model weights** and only trains a tiny set of **low‑rank matrices** inserted into specific layers (e.g., `q_proj`, `v_proj`).

This gives you:
- 💾 *Small storage* — adapters are only a few MB  
- 🚀 *Fast training* — only a small fraction of params updated  
- 🔄 *Model safety* — base model stays intact  
- 🔌 *Plug & Play* — you can enable or disable adapters at inference

#### 🔹 Typical LoRA config used in this repo
	r = 8 or 16 (low-rank)
	lora_alpha = 16 or 32

### 4 Setup (virtualenv / dependencies)

#### 🔹 Create virtual environment (recommended)
````
python3 -m venv .venv
source .venv/bin/activate
````
#### 🔹 Install dependencies
````
pip install -r requirements.txt
````

##### Notes for Mac M1/M2/M3:
	- Use the official PyTorch install instructions for MPS if needed.
	- bitsandbytes / CUDA quantization is not supported on MPS; this repo uses LoRA to keep memory use low on Mac.

#### Example requirements.txt (this repo’s base)
````
torch
transformers
datasets
peft
accelerate
flask
sentencepiece
````

### 5 Training (train.py)

Purpose
-	Attaches LoRA adapters to the base microsoft/phi-2
-	Fine-tunes adapters on data/train_qa_500.jsonl
-	Saves adapters to models '/finetuned_phi2_lora_500/'

Key hyperparameters (safe defaults for Mac)
-	learning_rate = 1e-5 or 2e-5
-	num_train_epochs = 2
-	per_device_train_batch_size = 1
-	gradient_accumulation_steps = 4
-	max_length = 512
-	lora_r = 8 (or 16)
-	lora_alpha = 16 (or 32)
-	lora_dropout = 0.05

Example training flow (what train.py does)
1.	Load tokenizer, set pad_token if missing:
````
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
````
2.	Load base model (float16/float32 depending on MPS support):
````
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16)
````
3.	Prepare model for LoRA and attach adapters:
````
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
````

4.	Tokenize dataset with a fixed max_length and padding="max_length".
5. Use Trainer with conservative TrainingArguments.
6. Save adapter with model.save_pretrained(OUTPUT_DIR).


Output

After success, models/finetuned_phi2_lora_500/ contains adapter files you can load at inference time.


### 6 Dataset format

data/train_qa_500.jsonl is newline-delimited JSON. Each line is an object with prompt and completion:
````
{"prompt":"You are an interview coach. Role: Technical Program Manager. Q: How do you manage cross-team dependencies? A:","completion":"Create a shared dependency tracker..."}
````

Important:
-	Keep prompt structure consistent to teach the model patterns.
-	Avoid identical completions repeated across all prompts (causes overfitting).
-	Add mixed examples (e.g., storytelling, summarization) if you want to preserve general generation.

### 7 Inference (infer.py)

Purpose 
-   Load base model + adapter
-   Generate role-aware responses using generate_answer(prompt, role, length)

Options at inference
-   Base model only:
````
base = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
````

- 	Base + Adapter:
````
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
model = PeftModel.from_pretrained(base, "models/finetuned_phi2_lora_500/")
````

-   Merge adapter permanently:
````
model = model.merge_and_unload()
model.save_pretrained("models/phi2-fused")
````

Generation parameters (recommended)
````
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)
````

-   Increase max_new_tokens if you want long outputs.
-   Use repetition_penalty and no_repeat_ngram_size to avoid looping.

### 8 Web app (app.py)

A minimal Flask app that:
-   Presents a role dropdown
-   Accepts a prompt
-	Calls generate_answer(prompt, role, length)
-   Returns the generated text

Running
````
python app.py
# open http://127.0.0.1:5000
````

Suggested UI features (implemented)
-	Role selector (TPM, PM, Software Engineer, Data Scientist, ML Engineer)
-	Length selector (short, medium, long)
-	Toggle: use base model vs use fine-tuned adapter


### 9 Running in PyCharm
1. Open project in PyCharm. 
2. Configure project interpreter to .venv or a Python with dependencies installed.
3. Create Run/Debug configurations:
   - Script path: train.py (or app.py / infer.py)
4. Run and monitor logs in PyCharm console.


### 10 Makefile & helper scripts

Makefile targets:
-	make train — run training in venv
-	make serve — run Flask app
-	make docker-run — build & run Docker Compose

./scripts/train.sh and ./scripts/serve.sh wrap environment activation and run commands for convenience.


### 11 Docker (CPU-only / Mac-safe)

Dockerfile uses python:3.10-slim and installs packages in requirements.txt. The image runs app.py.

docker-compose.yml builds and mounts the repo; it does not enable GPU.

Build & run
````
docker-compose up --build
# or
docker build -t custom-llm-with-lora .
docker run -p 5000:5000 custom-llm-with-lora
````

### 12 Troubleshooting & FAQ

Q: Tokenizer has no pad_token

A: Set it:

````
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
````

Q: ArrowInvalid: expected length X but got Y

A: Ensure tokenization returns fixed-length arrays if mapping directly. Use padding="max_length", max_length=512 when tokenizing inside .map() or use DataCollatorWithPadding to pad at batch time.

Q: Generation too short even with high max_new_tokens

A: Likely caused by fine-tuning on repetitive completions. Fix by:
-	Adding diverse training examples (stories, summarization)
-	Using adapters via LoRA and load base model for general tasks

Q: bitsandbytes / 4-bit warnings on Mac

A: bitsandbytes requires CUDA. Do not use on MPS. Use float32/float16 with LoRA on Mac.

---

## 📝 License

This project is licensed under the **MIT License**.
