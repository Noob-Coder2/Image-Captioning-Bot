from app import app
from flask import request, jsonify
from werkzeug.utils import secure_filename
import os
from app.inference import caption_image

os.makedirs("uploads", exist_ok=True)

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
