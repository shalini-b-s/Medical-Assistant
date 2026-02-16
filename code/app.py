from flask import Flask, request, jsonify, render_template
from model import llama_hf_response, llama_ibm_response, granite_response, mistral_response
from augmented_prompt import rag_prompt, format_context
from retriever import retriever
import time
import warnings 
warnings.filterwarnings(action= 'ignore')

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods = ['POST'])
def generate():
    data = request.json
    user_message = data.get('message')
    model = data.get('model')

    if not user_message or not model: 
        return jsonify({"error": "Missing message or model selection"}), 400
    
    # system_prompt = "You are an AI assistant helping with customer inquiries. Provide a helpful and concise response."

    # start_time = time.time()

    relevant_child_chunks, distances = retriever.get_relevant_documents(user_message)
    context = format_context(relevant_child_chunks)

    try:
        if model == 'llama_hf':
            result = llama_hf_response(context, user_message)
        elif model == 'llama_ibm':
            result = llama_ibm_response(context, user_message)
        elif model == 'granite':
            result = granite_response(context, user_message)
        elif model == 'mistral':
            result = mistral_response(context, user_message)
        else:
            return jsonify({'error': "Invalid model selection"}), 400
        
        # Note: result here is likely a BaseMessage from LangChain, check its attributes
        return jsonify({'content': result.content})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug = True)