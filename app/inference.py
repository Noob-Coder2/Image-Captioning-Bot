import torch
from utils.preprocess import preprocess_image
from models.model import load_fine_tuned_model, generate_caption

def caption_image(image_path):
    try:
        # Initialize all model components
        clip_model, clip_processor, gpt_model, tokenizer, clip_to_gpt = load_fine_tuned_model()
        
        # Verify all components are initialized
        if None in [clip_model, clip_processor, gpt_model, tokenizer, clip_to_gpt]:
            return "Error: Failed to initialize model components"
        
        # Preprocess the image
        image = preprocess_image(image_path, clip_processor)
        if image is None:
            return "Error: Failed to preprocess image"
        
        with torch.no_grad():
            # Get image features and ensure they're on the correct device
            image_features = clip_model.get_image_features(image["pixel_values"].to(clip_model.device))
            
            try:
                # First try using generate_caption
                caption = generate_caption(image_features, gpt_model, tokenizer, clip_to_gpt)
            except Exception as e:
                # Fallback to GPT model's generate method if generate_caption fails
                print(f"Warning: generate_caption failed, falling back to GPT generate: {str(e)}")
                try:
                    # Use GPT's standard generate method with proper parameters
                    input_ids = tokenizer.encode("<SOS>", return_tensors="pt").to(gpt_model.device)
                    outputs = gpt_model.generate(
                        input_ids=input_ids,
                        max_length=50,
                        num_beams=5,
                        early_stopping=True
                    )
                    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
                except Exception as e:
                    return f"Error: Both caption generation methods failed - {str(e)}"
            
            return caption
            
    except Exception as e:
        return f"Error generating caption: {str(e)}"
