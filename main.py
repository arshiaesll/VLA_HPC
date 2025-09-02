

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
os.environ["HF_Home"] = "/work/aeslami/.cache"


def run_model():

    model_name = "microsoft/Phi-3.5-mini-instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )

    model = AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    print("here")
    run_model()