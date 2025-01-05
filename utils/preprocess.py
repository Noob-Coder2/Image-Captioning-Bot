from PIL import Image
from transformers import CLIPProcessor
import torch

# Load processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def preprocess_image(image_paths):
    """
    Preprocess single image or batch of images for CLIP model.
    
    Args:
        image_paths: Can be either a single image path (str) or a list of image paths
        
    Returns:
        Tensor of preprocessed images with shape (batch_size, 3, 224, 224)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert single path to list for uniform processing
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    try:
        # Load and convert all images to RGB
        images = [Image.open(path).convert("RGB") for path in image_paths]
        
        # Process batch of images
        inputs = processor(images=images, return_tensors="pt")
        return inputs['pixel_values'].to(device)
        
    except Exception as e:
        print(f"Error processing images: {e}")
        return None
