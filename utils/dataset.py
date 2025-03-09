import pandas as pd
import torch
from torch.utils.data import IterableDataset
from PIL import Image
import requests
import logging
from transformers import CLIPProcessor, GPT2Tokenizer
import time

logger = logging.getLogger(__name__)

def load_image(url, max_retries=3, timeout=5):
    """Download image with retries and exponential backoff."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=timeout, headers=headers)
            response.raise_for_status()
            return Image.open(response.raw).convert("RGB")
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"Failed to download {url}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None

class StreamingImageCaptionDataset(IterableDataset):
    def __init__(self, csv_file, processor, tokenizer, chunk_size=1000):
        self.csv_file = csv_file
        self.processor = processor
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        # Add special <image> token
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
        self.image_token_id = self.tokenizer.convert_tokens_to_ids('<image>')

    def __iter__(self):
        reader = pd.read_csv(self.csv_file, chunksize=self.chunk_size)
        for chunk in reader:
            for _, row in chunk.iterrows():
                url = row['url']
                caption = row['caption']
                try:
                    img = load_image(url)
                    if img is None:
                        continue
                    # Preprocess image
                    image = self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
                    # Tokenize caption and prepend <image> token
                    caption_ids = self.tokenizer.encode(caption, return_tensors="pt").squeeze(0)
                    input_ids = torch.cat([torch.tensor([self.image_token_id]), caption_ids])
                    yield image, input_ids
                except Exception as e:
                    logger.warning(f"Error processing {url}: {e}")
                    continue

