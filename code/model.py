from langchain_ibm import ChatWatsonx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config import params, url, api_key, project_id, LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID, HF_LLAMA_MODEL_ID, HF_API_KEY
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class AIResponse(BaseModel):
    summary: str = Field(description= "summary of the user's message")
    sentiment: int = Field(description=  "Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description = "Suggested response to the user")

json_parser = JsonOutputParser(pydantic_object = AIResponse)

# Dictionary to hold initialized models
_models = {}

def get_model(model_type):
    """Lazily initialize and return the requested model."""
    if model_type in _models:
        return _models[model_type]
    
    if model_type == 'llama_hf':
        if not HF_API_KEY:
            raise ValueError("HUGGINGFACE_HUB_TOKEN is missing in .env file.")
        model = ChatOpenAI(
            model=HF_LLAMA_MODEL_ID,
            base_url="https://router.huggingface.co/v1",
            api_key=HF_API_KEY
        )
    elif model_type in ('granite', 'mistral', 'llama_ibm'):
        if not all([url, api_key, project_id]):
            missing = [k for k, v in {'url': url, 'api_key': api_key, 'project_id': project_id}.items() if not v]
            raise ValueError(f"Watsonx credentials missing: {', '.join(missing)}")
        
        if model_type == 'granite':
            model_id = GRANITE_MODEL_ID
        elif model_type == 'mistral':
            model_id = MISTRAL_MODEL_ID
        else: # llama_ibm
            model_id = LLAMA_MODEL_ID
            
        model = ChatWatsonx(
            model_id=model_id, 
            url=url, 
            apikey=api_key, 
            project_id=project_id, 
            params=params
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    _models[model_type] = model
    return model

# Prompt templates
llama_hf_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "user_prompt"]
)

llama_ibm_template = PromptTemplate(
    template="<|system|>{system_prompt}\n<|user|>{user_prompt}\n<|assistant|>",
    input_variables=["system_prompt", "user_prompt"]
)

granite_template = PromptTemplate(
    template="<|system|>{system_prompt}\n<|user|>{user_prompt}\n<|assistant|>",
    input_variables=["system_prompt", "user_prompt"]
)

mistral_template = PromptTemplate(
    template="<s>[INST]{system_prompt}\n{user_prompt}[/INST]",
    input_variables=["system_prompt", "user_prompt"]
)

def get_ai_response(model_type, template, system_prompt, user_prompt):
    model = get_model(model_type)
    chain = template | model
    return chain.invoke({'system_prompt': system_prompt, 'user_prompt': user_prompt, 'format_prompt':json_parser.get_format_instructions()})

def llama_hf_response(system_prompt, user_prompt):
    return get_ai_response('llama_hf', llama_hf_template, system_prompt, user_prompt)

def llama_ibm_response(system_prompt, user_prompt):
    return get_ai_response('llama_ibm', llama_ibm_template, system_prompt, user_prompt)

def granite_response(system_prompt, user_prompt):
    return get_ai_response('granite', granite_template, system_prompt, user_prompt)

def mistral_response(system_prompt, user_prompt):
    return get_ai_response('mistral', mistral_template, system_prompt, user_prompt)
