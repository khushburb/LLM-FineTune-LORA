# src/infer.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_DIR = "finetuned_phi2_lora_500"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

def answer(question: str):
    prompt = f"You are an interview coach. Q: {question} A:"
    res = generator(prompt, do_sample=True, temperature=0.2, top_p=0.95, num_return_sequences=1)
    return res[0]["generated_text"].split("A:")[-1].strip()

if __name__ == "__main__":
    while True:
        q = input("Question (or 'exit'): ")
        if q.lower()=="exit": break
        print("\nAnswer:\n", answer(q))
