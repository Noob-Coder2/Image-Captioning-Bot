import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules after setting up the path
from app.inference import caption_image

# Create Flask app
app = Flask(__name__, static_folder='../static')
CORS(app)  # Enable CORS for all routes

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/generate_caption', methods=['POST'])
def generate_caption_api():
    try:
        image_file = request.files['image']
        image_path = os.path.join("uploads", secure_filename(image_file.filename))
        image_file.save(image_path)

        # Check if the model is already trained
        model_path = "models/fine_tuned_clip"
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not trained yet. Please train the model first."}), 400

        caption = caption_image(image_path)
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
