import torch
from utils.preprocess import preprocess_image
from models.model import load_fine_tuned_model, generate_caption, initialize_model
from transformers import GPT2LMHeadModel

def caption_image(image_path):
    """
    Generate a caption for an image, with a fallback mechanism if the primary method fails.

    Args:
        image_path (str): Path to the image file or URL.

    Returns:
        str: Generated caption or error message.
    """
    try:
        # Initialize primary model components
        model, clip_processor, tokenizer = load_fine_tuned_model()  # Returns ImageConditionedGPT2, processor, tokenizer
        clip_model = initialize_model()  # CLIPModel
        
        # Verify initialization
        if None in [model, clip_processor, tokenizer, clip_model]:
            return "Error: Failed to initialize model components"
        
        # Preprocess the image
        image = preprocess_image(image_path)  # Assuming this returns a tensor
        if image is None:
            return "Error: Failed to preprocess image"
        
        with torch.no_grad():
            # Get image features
            image_features = clip_model.get_image_features(image.to(clip_model.device))
            
            # image_features = clip_model.get_image_features(image['pixel_values'].to(clip_model.device))
            
            try:
                # Primary method: Use ImageConditionedGPT2 with <image> token
                caption = generate_caption(image_features, model, tokenizer)
                return caption
            except Exception as e:
                print(f"Warning: Primary caption generation failed, falling back to vanilla GPT-2: {str(e)}")
                
                # Fallback: Use a vanilla GPT-2 model
                try:
                    # Load vanilla GPT-2 as fallback
                    fallback_model = GPT2LMHeadModel.from_pretrained("gpt2").to(model.device)
                    # Simple projection of image features to GPT-2 embedding space
                    proj_layer = torch.nn.Linear(512, fallback_model.config.n_embd).to(model.device)
                    gpt_input = proj_layer(image_features).unsqueeze(1)  # [batch, 1, n_embd]
                    # Start with a dummy input and use embeddings
                    input_ids = tokenizer.encode(" ", return_tensors="pt").to(model.device)  # Minimal text input
                    outputs = fallback_model.generate(
                        input_ids=input_ids,
                        inputs_embeds=gpt_input,
                        max_length=50,
                        num_beams=5,
                        early_stopping=True
                    )
                    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return caption
                except Exception as e:
                    return f"Error: Both caption generation methods failed - {str(e)}"
            
    except Exception as e:
        return f"Error generating caption: {str(e)}"

