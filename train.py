import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------------------------
# 1️⃣ Basic Configuration
# -------------------------------
MODEL_NAME = "microsoft/phi-2"
DATA_PATH = "data/train_500.jsonl"  # path to your dataset
OUTPUT_DIR = "./phi2_finetuned_lora_500"

# Detect device (Macs: MPS; others: CUDA/CPU)
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {DEVICE}")

# -------------------------------
# 2️⃣ Load Model and Tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load model in 4-bit (for memory efficiency on Mac)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

# -------------------------------
# 3️⃣ Configure LoRA
# -------------------------------
lora_config = LoraConfig(
    r=8,                      # rank of LoRA update matrices
    lora_alpha=16,            # scaling factor
    target_modules=["q_proj", "v_proj"],  # specific transformer modules to train
    lora_dropout=0.05,        # slight dropout for stability
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------------
# 4️⃣ Load Dataset
# -------------------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_prompt(example):
    """
    Combine prompt and completion into a single text sequence for causal LM training.
    """
    text = f"{example['prompt']} {example['completion']}"
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(format_prompt, batched=True, remove_columns=dataset.column_names)

# -------------------------------
# 5️⃣ Training Configuration
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,         # 🔸 smaller learning rate to preserve base model
    num_train_epochs=2,         # 🔸 fewer epochs to avoid overfitting
    fp16=True if DEVICE != "cpu" else False,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

# -------------------------------
# 6️⃣ Data Collator
# -------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -------------------------------
# 7️⃣ Trainer Setup
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# -------------------------------
# 8️⃣ Start Fine-tuning
# -------------------------------
trainer.train()
trainer.save_model(OUTPUT_DIR)

print(f"✅ Fine-tuning complete! Model saved to {OUTPUT_DIR}")