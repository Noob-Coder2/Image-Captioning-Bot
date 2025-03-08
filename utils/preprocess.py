import logging
from time import sleep
from PIL import Image
from transformers import CLIPProcessor
import torch
import requests
from io import BytesIO
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

# Load processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Configuration
MAX_RETRIES = 1
INITIAL_TIMEOUT = 2
BACKOFF_FACTOR = 2
HEADERS = {
    'User-Agent': 'Chrome/91.0.4472.124' #'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def is_valid_image_url(url):
    """Validate image URL format and scheme."""
    parsed = urlparse(url)
    return (parsed.scheme in ('http', 'https') and 
            parsed.netloc and 
            any(parsed.path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']))

def download_image(url, retries=MAX_RETRIES, timeout=INITIAL_TIMEOUT):
    """Download image from URL with retries and exponential backoff."""
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                stream=True,
                timeout=timeout
            )
            response.raise_for_status()
            return BytesIO(response.content)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            sleep_time = BACKOFF_FACTOR ** attempt
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time} seconds...")
            sleep(sleep_time)
    return None

def preprocess_image(image_paths):
    """
    Preprocess single image or batch of images for CLIP model.
    Handles both local file paths and URLs.
    
    Args:
        image_paths: Can be either a single image path (str) or a list of image paths
        
    Returns:
        Tensor of preprocessed images with shape (batch_size, 3, 224, 224)
        or None if processing fails
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert single path to list for uniform processing
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    try:
        images = []
        for path in image_paths:
            try:
                # Check if path is a URL
                if urlparse(path).scheme in ('http', 'https'):
                    if not is_valid_image_url(path):
                        logger.error(f"Invalid image URL format: {path}")
                        continue
                    
                    img_data = download_image(path)
                    if img_data is None:
                        logger.error(f"Failed to download image from URL: {path}")
                        continue
                        
                    img = Image.open(img_data).convert("RGB")
                else:
                    # Local file path
                    img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.error(f"Error processing image {path}: {e}")
                continue
        
        if not images:
            logger.error("No valid images were processed")
            return None
            
        # Process batch of images
        inputs = processor(images=images, return_tensors="pt")
        return inputs['pixel_values'].to(device)
        
    except Exception as e:
        logger.error(f"Error in preprocess_image: {e}")
