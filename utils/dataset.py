import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


def load_captions(file_path):
    """
    Loads captions and image paths from a .tsv file.
    The file should be in format: caption \t image_url

    Args:
        file_path (str): Path to the .tsv file.

    Returns:
        tuple: (list of image paths, list of captions)

    Raises:
        ValueError: If the file format is invalid
    """
    try:
        # Read the TSV file without headers
        data = pd.read_csv(file_path, sep='\t', header=None)
        
        # Verify we have exactly two columns
        if len(data.columns) != 2:
            raise ValueError("TSV file must have exactly two columns: caption and image_url")
            
        # Extract captions and image paths
        captions = data[0].tolist()
        image_paths = data[1].tolist()
        
        return image_paths, captions
        
    except Exception as e:
        raise ValueError(f"Error loading captions from {file_path}: {str(e)}")


class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, processor, tokenizer):
        """
        Dataset class for loading image-caption pairs.

        Args:
            image_paths (list): List of paths to the image files.
            captions (list): List of corresponding captions.
            processor (Callable): Processor to preprocess images.
            tokenizer (Callable): Tokenizer to preprocess captions.
        """
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves a single image-caption pair.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Preprocessed image and caption.
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        caption = self.tokenizer(self.captions[idx], return_tensors="pt", padding=True, truncation=True)
        return inputs['pixel_values'].squeeze(0), caption['input_ids'].squeeze(0)
