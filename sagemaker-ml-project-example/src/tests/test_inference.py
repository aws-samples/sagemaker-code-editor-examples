from sagemaker.huggingface import HuggingFacePredictor
import traceback

def predict(endpoint_name, prompt, parameters={}):
    """Make a prediction by using the SageMaker endpoint provided for a test case. Return True if the response contains an array
    with generated_text, False otherwise. Put everithing in a try/except block.
    Parameters:
    endpoint_name: Endpoint to invoke
    prompt: prompt to send to the model and part of the json for the predict as inputs
    parameters: parameters for the llm and part of the json for the predict as parameters. if None, or empty, don't add
                parameters in the json
    """
    try:
        predictor = HuggingFacePredictor(endpoint_name)
        if parameters:
            response = predictor.predict({"inputs": prompt, "parameters": parameters})
        else:
            response = predictor.predict({"inputs": prompt})
        if "generated_text" in response[0].keys():
            return True
        return False
    except Exception as e:
        stacktrace = traceback.format_exc()
        print(stacktrace)

        return False
