from app.train_pipeline import train_pipeline
import os
from google.colab import drive

if __name__ == "__main__":
    
    # File paths
    train_file = "uploads/Train_GCC-training.tsv"  # Updated to use forward slashes
    validation_file = "uploads/Validation_GCC-1.1.0-Validation.tsv"  # Updated to use forward slashes
    results_path = "static/results.json"
    model_save_path = "/content/drive/My Drive/models/fine_tuned_clip"  # Save to Google Drive

    # Check if the model already exists
    if os.path.exists(model_save_path):
        print(f"Model already exists at {model_save_path}. Skipping training.")
    else:
        # Run the training pipeline
        train_pipeline(train_file, validation_file, results_path, model_save_path)
