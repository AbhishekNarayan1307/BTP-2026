
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

# Use Apple's GPU when available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded on: {device}")


messages = [
    {"role": "user", "content": "What is the capital of India?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

# Move input tensors to the selected device
inputs = {key: value.to(device) for key, value in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print("\nAnswer:", answer)

