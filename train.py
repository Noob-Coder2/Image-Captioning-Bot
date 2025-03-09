from app.train_pipeline import train_pipeline
import os
#from google.colab import drive

if __name__ == "__main__":
    # File paths (adjusted for Kaggle or local use)
    train_file = "/kaggle/input/image-captioning-bot-project/Train_GCC-processed.csv"  # Update as per your dataset path
    validation_file = "/kaggle/input/image-captioning-bot-project/Validation_GCC-1.1.0-Validation.tsv"  # Optional
    results_path = "/kaggle/working/static/results.json"  # Writable directory
    model_save_path = "/kaggle/working/models/fine_tuned_model"  # Writable directory

    # Ensure output directories exist
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Run the pipeline
    train_pipeline(train_file, validation_file, results_path, model_save_path)