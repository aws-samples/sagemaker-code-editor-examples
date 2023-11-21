# This is the script that will be used in the inference container
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir):
    """
    Load the model and tokenizer for inference
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(device)

    model.generate

    return {"model": model, "tokenizer": tokenizer}


def predict_fn(input_data, model_dict):
    """
    Make a prediction with the model
    """
    text = input_data.pop("inputs")
    parameters = input_data.pop("parameters", dict())

    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]

    # Parameters may or may not be passed
    input_ids = tokenizer(
        text, truncation=True, padding="longest", return_tensors="pt"
    ).input_ids.to(device)

    outputs = model.generate(
        inputs=input_ids, 
        max_length=parameters["max_new_tokens"],
        do_sample=parameters["do_sample"] if "do_sample" in parameters else True,
        temperature=parameters["temperature"] if "temperature" in parameters else 1.0,
        top_p=parameters["top_p"] if "top_p" in parameters else 0.9
    )
    gen_text = tokenizer.batch_decode(outputs)[0]

    return json.dumps([{
        "generated_text": gen_text
    }])


def input_fn(request_body, request_content_type):
    """
    Transform the input request to a dictionary
    """
    return json.loads(request_body)
