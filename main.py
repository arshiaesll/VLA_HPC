

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
os.environ["HF_Home"] = "/work/aeslami/.cache"
print("Loaded the libraries!")

def run_model():

    model_name = "microsoft/Phi-3.5-mini-instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model = model,
        tokenizer=tokenizer
    )

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]    

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "do_sample": False,
        "use_cache": False
    }

    msg = pipe(messages, **generation_args)
    print(msg)


if __name__ == "__main__":
    run_model()
    print("Done")