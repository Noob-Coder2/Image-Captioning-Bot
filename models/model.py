import torch
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer
from models.clip_gpt_bridge import CLIP2GPT

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    return clip_model

# Load models and tokenizer
def load_fine_tuned_model(path="models/fine_tuned_model"):
    """
    Initializes and returns the CLIP model, processor, GPT2 model, tokenizer, and CLIP2GPT bridge.
    """
    # Load CLIP model
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(path).to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    # Load GPT2 model
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load CLIP2GPT bridge
    clip_to_gpt = CLIP2GPT(clip_dim=512, gpt_dim=gpt_model.config.n_embd).to(device)
    
    return fine_tuned_model, clip_processor, gpt_model, tokenizer, clip_to_gpt


def train_model(model, train_images, train_captions, epochs=5, learning_rate=1e-4):
    """
    Trains the model using the provided training data.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_images (torch.Tensor): Preprocessed images for training.
        train_captions (torch.Tensor): Tokenized captions for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        torch.nn.Module: Trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # loss_fn = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for image, caption in list(zip(train_images, train_captions)):
            optimizer.zero_grad()
            
            # Move inputs to device
            image = image.to(device)
            caption = caption.to(device)
            
            # Forward pass
            outputs = model(image, labels=caption)
            loss = outputs.loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
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


def generate_caption(image_embedding, gpt_model, tokenizer, clip_to_gpt, fine_tuned_model=None):
    """
    Generates a caption for the provided image embedding using either the base or fine-tuned model.

    Args:
        image_embedding (torch.Tensor): The embedding of the image.
        gpt_model (GPT2LMHeadModel): Base GPT2 model.
        tokenizer (GPT2Tokenizer): Tokenizer for text processing.
        clip_to_gpt (CLIP2GPT): CLIP-GPT bridge model.
        fine_tuned_model (GPT2LMHeadModel, optional): Fine-tuned GPT2 model. Defaults to None.

    Returns:
        str: Generated caption.
    """
    model_to_use = fine_tuned_model if fine_tuned_model is not None else gpt_model
    gpt_input = clip_to_gpt(image_embedding)
    input_ids = tokenizer.encode(gpt_input, return_tensors="pt").to(device)
    outputs = model_to_use.generate(input_ids=input_ids, max_length=50, num_beams=5, early_stopping=True)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
