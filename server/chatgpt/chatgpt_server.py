from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the ChatGPT Flask Server!"


@app.route('/generate', methods=['POST'])
def chat():
    user_input = request.json.get('text')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    try:
        response = chat_completion(user_input)
        return jsonify({'result': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
def chat_completion (
        prompt,
        model=os.environ.get('Model'),
        max_tokens=600,
        stop_sequence=None
):
    try:
        # Create a completions using the chat
        response = openai.ChatCompletion.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        print(response)
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""
    
if __name__ == '__main__':
    openai.api_key=os.environ.get('Key')
    app.run(host='0.0.0.0', debug=True)  # Host on all available interfaces, enable debug mode