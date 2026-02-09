import os 
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams


load_dotenv()


url = os.getenv('url')
api_key = os.getenv('api_key')
project_id = os.getenv('project_id')

credentials = Credentials(url= url, api_key= api_key)

params = {GenParams.TEMPERATURE: 0.5, GenParams.MAX_NEW_TOKENS: 128}

LLAMA_MODEL_ID = "meta-llama/llama-3-2-11b-vision-instruct"
GRANITE_MODEL_ID = "ibm/granite-3-8b-instruct"
MISTRAL_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"

HF_LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct:novita"
HF_API_KEY = os.getenv('HUGGINGFACE_HUB_TOKEN')