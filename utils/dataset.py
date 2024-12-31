from torch.utils.data import Dataset
from PIL import Image

class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, processor, tokenizer):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        caption = self.tokenizer(self.captions[idx], return_tensors="pt", padding=True, truncation=True)
        return inputs['pixel_values'].squeeze(0), caption['input_ids'].squeeze(0)
