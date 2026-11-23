
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 300

# Model objects are initially unloaded
tokenizer = None
model = None
device = None


def load_model():
    """
    Load the Qwen model and tokenizer.
    Load only once per Python process.
    """
    global tokenizer, model, device

    if model is not None:
        return tokenizer, model, device

    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading Qwen model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    # Select the available device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully on: {device}")

    return tokenizer, model, device


def generate_response(prompt, max_new_tokens=MAX_NEW_TOKENS):
    """
    Generate a response from Qwen using the supplied prompt.
    """

    tokenizer, model, device = load_model()

    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return answer.strip()
