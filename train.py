from app.train_pipeline import train_pipeline

if __name__ == "__main__":
    # File paths
    train_file = "Train_GCC-training.tsv"
    validation_file = "Validation_GCC-1.1.0-Validation.tsv"
    results_path = "static/results.json"  # Consider checking if the "static" directory exists before saving to avoid runtime errors.
    model_save_path = "models/fine_tuned_clip"  # Ensure "models" directory exists or handle directory creation dynamically.

    # Run the training pipeline
    train_pipeline(train_file, validation_file, results_path, model_save_path)  # Add error handling to catch any exceptions raised during training.
