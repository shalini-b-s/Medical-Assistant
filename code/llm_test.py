from model import llama_hf_response, llama_ibm_response, granite_response, mistral_response
import warnings
warnings.filterwarnings(action= 'ignore')

def call_all_models(system_prompt, user_prompt):
    models = {
        'Llama HF': llama_hf_response,
        'Llama IBM': llama_ibm_response,
        'Granite': granite_response,
        'Mistral': mistral_response
    }

    for name, func in models.items():
        print(f"\n--- Testing {name} ---")
        try:
            result = func(system_prompt, user_prompt)
            print(f'{name} response: \n', result.content)
        except Exception as e:
            print(f'Error calling {name}: {e}')

call_all_models("You are a helpful assistant who provides concise and accurate answers", 'What is the capital of Canada? Tell me a cool fact about it as well')
