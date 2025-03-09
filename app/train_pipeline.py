import os
import torch
from torch.utils.data import DataLoader
from utils.dataset import StreamingImageCaptionDataset
from models.model import train_model, save_model, initialize_model
from models.clip_gpt_bridge import ImageConditionedGPT2
from transformers import GPT2Config, CLIPProcessor, GPT2Tokenizer
from app.evaluation import calculate_bleu, calculate_rouge, save_results

def train_pipeline(train_file, validation_file=None, results_path="static/results.json", model_save_path="models/fine_tuned_model"):
    """
    Trains the Image Captioning model on the provided dataset with checkpoints and evaluates if validation data is provided.

    Args:
        train_file (str): Path to the training CSV/TSV file (e.g., Train_GCC-processed.csv).
        validation_file (str, optional): Path to the validation TSV file (e.g., Validation_GCC-1.1.0-Validation.tsv).
        results_path (str): Path to save evaluation results.
        model_save_path (str): Path to save the trained model.
    """
    # Check if model is already trained
    if os.path.exists(model_save_path):
        print(f"Model already trained and available at {model_save_path}. Skipping training.")
        return

    # Step 1: Initialize components
    print("Starting the training pipeline...")
    print("Initializing model components...")
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    clip_model = initialize_model()  # CLIPModel from models/model.py
    config = GPT2Config.from_pretrained("gpt2")
    model = ImageConditionedGPT2(config)
    model.tokenizer = tokenizer  # Attach tokenizer for <image> token
    model.resize_token_embeddings(len(tokenizer))

    # Step 2: Setup training dataset and dataloader
    print("Loading and preprocessing training dataset...")
    train_dataset = StreamingImageCaptionDataset(train_file, processor, tokenizer, chunk_size=1000)
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=4)  # Kaggle: 4 vCPUs

    # Step 3: Train the model with checkpoints
    print("Training the model...")
    checkpoint_path = os.path.join(os.path.dirname(model_save_path), "checkpoint.pth")
    trained_model = train_model(
        model,
        train_dataloader,
        clip_model,
        total_steps=10000,  # Adjust based on your needs
        save_interval=1000,
        checkpoint_path=checkpoint_path
    )

    # Step 4: Save the trained model
    save_model(trained_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Step 5: Evaluate if validation file is provided
    if validation_file:
        print("Evaluating the model...")
        # Load validation data (assuming same format as train_file)
        val_dataset = StreamingImageCaptionDataset(validation_file, processor, tokenizer, chunk_size=1000)
        val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=4)
        
        # Preprocess validation data for evaluation
        validation_images = []
        validation_captions = []
        for batch in val_dataloader:
            images, input_ids = batch
            with torch.no_grad():
                image_features = clip_model.get_image_features(images.to(clip_model.device))
            validation_images.extend(image_features.cpu())
            validation_captions.extend([tokenizer.decode(ids[1:], skip_special_tokens=True) for ids in input_ids])  # Skip <image>
            if len(validation_images) >= 100:  # Limit for evaluation
                break

        # Calculate metrics
        bleu_scores = calculate_bleu(trained_model, validation_images, validation_captions)
        rouge_scores = calculate_rouge(trained_model, validation_images, validation_captions)

        # Save results
        results = {"BLEU": bleu_scores, "ROUGE": rouge_scores}
        save_results(results, results_path)
        print(f"Evaluation results saved to {results_path}")
