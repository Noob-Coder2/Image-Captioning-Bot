from PIL import Image
from transformers import CLIPProcessor
import torch

# Load processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
