from utils.preprocess import preprocess_image
from models.model import model, clip_to_gpt, generate_caption

def caption_image(image_path):
    image = preprocess_image(image_path)
    if image is not None:
        with torch.no_grad():
            image_embedding = model.get_image_features(image)
            caption = generate_caption(image_embedding)
            return caption
    return "Image processing failed."
