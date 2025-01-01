import os
from utils.dataset import load_captions, preprocess_images
from models.model import initialize_model, train_model, save_model
from app.evaluation import calculate_bleu, calculate_rouge, save_results

def train_pipeline(train_file, validation_file, results_path, model_save_path):
    # Check if model is already trained
    if os.path.exists(model_save_path):
        print(f"Model already trained and available at {model_save_path}. Skipping training.")
        return

    # Step 1: Load and preprocess datasets
    print("Loading and preprocessing datasets...")
    train_data = load_captions(train_file, split="train")
    validation_data = load_captions(validation_file, split="validation")

    train_images, train_captions = preprocess_images(train_data)
    validation_images, validation_captions = preprocess_images(validation_data)

    # Step 2: Initialize the model
    print("Initializing the model...")
    model, clip_processor = initialize_model()

    # Step 3: Train the model
    print("Training the model...")
    trained_model = train_model(model, train_images, train_captions, epochs=5)

    # Save the trained model
    save_model(trained_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Step 4: Evaluate the model
    print("Evaluating the model...")
    bleu_scores = calculate_bleu(trained_model, validation_images, validation_captions)
    rouge_scores = calculate_rouge(trained_model, validation_images, validation_captions)

    # Step 5: Save evaluation results
    print("Saving evaluation results...")
    results = {
        "BLEU": bleu_scores,
        "ROUGE": rouge_scores,
    }
    save_results(results, results_path)
    print(f"Evaluation results saved to {results_path}")
