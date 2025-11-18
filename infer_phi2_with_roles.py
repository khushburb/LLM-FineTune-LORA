import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1️⃣ Load your fine-tuned model (update path if needed)
MODEL_PATH = "finetuned_phi2_lora_500"  # Replace with your actual fine-tuned model path
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔹 Using device: {DEVICE}")

# 2️⃣ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Ensure pad token is defined (important for Phi or models without padding tokens)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# 3️⃣ Build a generation pipeline for easy inference
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE != "cpu" else -1
)

# 4️⃣ Optional: role-aware system prompts
role_prompts = {
    "Technical Program Manager": "You are an experienced Technical Program Manager. Provide structured, leadership-oriented answers.",
    "Software Engineer": "You are a detail-oriented Software Engineer. Focus on clarity, performance, and maintainable design.",
    "Product Manager": "You are a strategic Product Manager. Think in terms of user value, KPIs, and market fit.",
    "Data Scientist": "You are an analytical Data Scientist. Focus on data-driven insights and statistical reasoning.",
    "Machine Learning Engineer": "You are a practical Machine Learning Engineer. Focus on ML pipelines, optimization, and deployment best practices."
}


# 5️⃣ Inference helper function
def generate_answer(question: str, role: str, max_tokens=10000):
    """
    Given a role and a question, build a structured prompt and generate a response.
    """
    if role not in role_prompts:
        print(f"⚠️ Unknown role '{role}', defaulting to general context.")
        system_prompt = "You are a professional AI assistant."
    else:
        system_prompt = role_prompts[role]

    # Construct the input for generation
    prompt = f"{system_prompt}\n\nQ: {question}\nA:"

    # Generate
    output = generator(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]

    # Extract only the answer portion
    if "A:" in output:
        answer = output.split("A:", 1)[-1].strip()
    else:
        answer = output.strip()

    return answer


# 6️⃣ Run a few examples interactively
if __name__ == "__main__":
    print("\n🚀 Role-Aware Model Inference\n")

    while True:
        role = input(
            "\nEnter Role (e.g., 'Technical Program Manager', 'Software Engineer', or 'quit' to exit): ").strip()
        if role.lower() == "quit":
            break
        question = input("Enter your question: ").strip()

        print("\n🔹 Generating answer...\n")
        answer = generate_answer(question, role)
        print(f"💬 {answer}\n")