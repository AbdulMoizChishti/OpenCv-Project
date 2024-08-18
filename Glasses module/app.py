from flask import Flask, request, jsonify
import subprocess
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/run-script', methods=['POST'])
def run_script():
    result = subprocess.run(['python', 'C:/Users/STARIZ.PK/Desktop/Saad project/Glasses module/new.py'], capture_output=True, text=True)
    return jsonify(result.stdout)

if __name__ == '__main__':
    app.run(debug=True)
