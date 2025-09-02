

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch



def run_model():

    model_name = "microsoft/Phi-3.5-mini-instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_device="auto",
        trust_remote_code=True
    )


if __name__ == "__main__":
    print("here")
    run_model()