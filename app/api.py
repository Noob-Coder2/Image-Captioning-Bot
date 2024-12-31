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
        caption = caption_image(image_path)
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
