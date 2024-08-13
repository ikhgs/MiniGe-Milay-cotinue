import os
import tempfile
from flask import Flask, request, jsonify
import google.generativeai as genai

# Configure the Google Generative AI SDK
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

# Dictionnaire pour stocker les contextes des utilisateurs
user_context = {}

def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(file_path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

@app.route('/api/process', methods=['POST'])
def process_image_and_prompt():
    if 'image' not in request.files or 'prompt' not in request.form or 'user_id' not in request.form:
        return jsonify({"error": "Image, prompt, and user_id are required."}), 400

    user_id = request.form['user_id']
    image = request.files['image']
    prompt = request.form['prompt']

    # Créer une conversation continue avec contexte
    chat_history = user_context.get(user_id, [])
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image_path = temp_file.name
        image.save(image_path)

        # Upload the image to Gemini
        file_uri = upload_to_gemini(image_path, mime_type=image.mimetype)

        # Ajouter l'image et le prompt à l'historique de la conversation
        chat_history.append({
            "role": "user",
            "parts": [
                file_uri,
                prompt,
            ],
        })

        # Create the chat session with the updated history
        chat_session = model.start_chat(history=chat_history)

        response = chat_session.send_message(prompt)

    # Clean up temporary file
    os.remove(image_path)

    # Mettre à jour le contexte de l'utilisateur avec la nouvelle réponse
    user_context[user_id] = chat_history + [{"role": "assistant", "parts": [response.text]}]

    return jsonify({"response": response.text})

@app.route('/api/query', methods=['GET'])
def query_prompt():
    user_id = request.args.get('user_id', default='default_user')
    prompt = request.args.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    # Obtenir l'historique existant pour l'utilisateur
    chat_history = user_context.get(user_id, [])
    
    # Ajouter la nouvelle question à l'historique
    chat_history.append({
        "role": "user",
        "parts": [prompt],
    })

    # Create the chat session with the updated history
    chat_session = model.start_chat(history=chat_history)

    response = chat_session.send_message(prompt)

    # Mettre à jour le contexte de l'utilisateur avec la nouvelle réponse
    user_context[user_id] = chat_history + [{"role": "assistant", "parts": [response.text]}]

    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
