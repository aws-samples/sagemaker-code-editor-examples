import json
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
import traceback

### Parameters ###
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
huggingface_container_version = "1.1.0"
inference_script = "./inference.py"

instance_count = 1
instance_type = "ml.g5.12xlarge"
number_of_gpu = 4
health_check_timeout = 300

def _get_huggingface_image():
    try:
        image_uri = get_huggingface_llm_image_uri(
            "huggingface",
            version=huggingface_container_version
        )
    except Exception as e:
        stacktrace = traceback.format_exc()
        print(stacktrace)

        raise e

def _deploy(image_uri):
    try:
        model = HuggingFaceModel(
            image_uri=image_uri,
            entry_point=inference_script,
            env={
                'HF_MODEL_ID': model_id, # path to where sagemaker stores the model
                'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
            }
        )

        model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            container_startup_health_check_timeout=health_check_timeout,
        )
    except Exception as e:
        stacktrace = traceback.format_exc()
        print(stacktrace)

        raise e

def main():
    image_uri = _get_huggingface_image()

    _deploy()

if __name__ == "__main__":
    main()
