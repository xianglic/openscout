from flask import Flask, request, jsonify
import subprocess
import shlex
import re

app = Flask(__name__)

def clean_text(txt):
    match = re.search(r'\n\n\n(.*?)(\[end of text\])', txt, re.DOTALL)

    if match:
        # match.group(1) contains the text captured by the first (and only) capture group in the regex
        result = match.group(1)
        return result.strip()  # strip() removes leading/trailing whitespace
    else:
        return "No match found"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data.get('text')
    input_text = input_text.replace('"', '')

    if input_text:
        command = f"./main -m ./llama-2-13b-ensemble-v6.Q5_K_S.gguf -p \"{input_text}\" -n 512 --n-gpu-layers 50 --temp 0"
        # command = f"./main -m ./Llama-2-13B-Ensemble-v6-GGUF/llama-2-13b-ensemble-v6.Q5_K_S.gguf -p \"{input_text}\" -n 512 --n-gpu-layers 50"

        try:
            output = subprocess.check_output(shlex.split(command), stderr=subprocess.STDOUT, text=True)
            output = clean_text(output)
            return jsonify({'result': output})
        except subprocess.CalledProcessError as e:
            return jsonify({'error': str(e), 'output': e.output}), 500
    else:
        return jsonify({'error': 'No input text provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)  # Host on all available interfaces, enable debug mode
