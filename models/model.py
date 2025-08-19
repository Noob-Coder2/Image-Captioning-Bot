import torch
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from models.clip_gpt_bridge import ImageConditionedGPT2
from torch.utils.data import DataLoader
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    return clip_model

# Load models and tokenizer
def load_fine_tuned_model(path="models/fine_tuned_model", conceptual_weights_path="uploads\conceptual_weights.pt"):
    """Initialize and return components without CLIP2GPT."""
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2")
    
    # Check for conceptual_weights.pt first
    if os.path.exists(conceptual_weights_path):
        print(f"Loading model with conceptual weights from {conceptual_weights_path}")
        fine_tuned_model = ImageConditionedGPT2(config).to(device)
        # Load the state dict directly
        state_dict = torch.load(conceptual_weights_path, map_location=device)
        fine_tuned_model.load_state_dict(state_dict)
    # Fallback to the original path
    elif os.path.exists(path):
        print(f"Loading model from {path}")
        fine_tuned_model = ImageConditionedGPT2.from_pretrained(path, config=config).to(device)
    # Create a new model if neither exists
    else:
        print("Initializing new model")
        fine_tuned_model = ImageConditionedGPT2(config).to(device)
    
    fine_tuned_model.tokenizer = tokenizer  # Attach for <image> token
    fine_tuned_model.resize_token_embeddings(len(tokenizer))
    return fine_tuned_model, clip_processor, tokenizer

def train_model(model, dataloader, clip_model, total_steps=10000, save_interval=1000, checkpoint_path="checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    clip_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    step = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        print(f"Resuming from step {step}")

    model.train()
    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break
            images, input_ids = batch
            images = images.to(device)
            input_ids = input_ids.to(device)

            # Compute image features
            with torch.no_grad():
                image_features = clip_model.get_image_features(images)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, image_features=image_features, labels=input_ids)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            step += 1
            print(f"Step {step}/{total_steps}, Loss: {loss.item():.4f}")

            # Save checkpoint
            if step % save_interval == 0:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved at step {step} to {checkpoint_path}")

    return model

def save_model(fine_tuned_model, path = "models/fine_tuned_model"):
    """
    Saves the model and processor to the specified path.

    Args:
        model (torch.nn.Module): The trained model to be saved.
        processor (transformers.PreTrainedProcessor): The processor to be saved.
        path (str): Directory to save the model and processor.
    """
    fine_tuned_model.save_pretrained(path)
    print(f"Model saved to {path}")


def generate_caption(image_embedding, model, tokenizer):
    """
    Generate a caption using ImageConditionedGPT2.

    Args:
        image_embedding (torch.Tensor): CLIP image features [batch, 512].
        model (ImageConditionedGPT2): The trained model.
        tokenizer (GPT2Tokenizer): Tokenizer with <image> token.

    Returns:
        str: Generated caption.
    """
    model.eval()
    with torch.no_grad():
        # Create input with <image> token
        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids('<image>')]).unsqueeze(0).to(model.device)
        # Forward pass with image features
        outputs = model.generate(
            input_ids=input_ids,
            image_features=image_embedding.to(model.device),
            max_length=50,
            num_beams=5,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
