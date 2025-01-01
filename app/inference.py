import torch
from utils.preprocess import preprocess_image
from models.model import load_trained_model

def caption_image(image_path):
    model, processor = load_trained_model("models/fine_tuned_clip")
    image = preprocess_image(image_path, processor)

    if image is not None:
        with torch.no_grad():
            image_features = model.clip_model.get_image_features(image["pixel_values"])
            caption = model.gpt_model.generate_text(image_features)
            return caption
    return "Image processing failed."
