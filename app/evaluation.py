import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

def calculate_bleu(model, validation_images, validation_captions):
    """
    Calculates BLEU scores for the model's generated captions against validation captions.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        validation_images (list): List of preprocessed validation images.
        validation_captions (list): List of ground truth captions.

    Returns:
        dict: BLEU scores (BLEU-1 to BLEU-4).
    """
    bleu_scores = {"BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": []}
    smoothing_function = SmoothingFunction().method1

    for image, ground_truth in zip(validation_images, validation_captions):
        # Generate caption
        with torch.no_grad():
            image = image.to(next(model.parameters()).device)
            generated_caption = model.generate_caption(image)

        # Tokenize captions
        generated_tokens = generated_caption.split()
        reference_tokens = [ground_truth.split()]

        # Compute BLEU scores
        bleu_scores["BLEU-1"].append(sentence_bleu(reference_tokens, generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function))
        bleu_scores["BLEU-2"].append(sentence_bleu(reference_tokens, generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function))
        bleu_scores["BLEU-3"].append(sentence_bleu(reference_tokens, generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function))
        bleu_scores["BLEU-4"].append(sentence_bleu(reference_tokens, generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function))

    # Average scores across all examples
    bleu_scores = {key: sum(values) / len(values) for key, values in bleu_scores.items()}
    return bleu_scores


def calculate_rouge(model, validation_images, validation_captions):
    """
    Calculates ROUGE scores for the model's generated captions against validation captions.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        validation_images (list): List of preprocessed validation images.
        validation_captions (list): List of ground truth captions.

    Returns:
        dict: ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    """
    rouge = rouge_scorer()
    rouge_scores = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": []}

    for image, ground_truth in zip(validation_images, validation_captions):
        # Generate caption
        with torch.no_grad():
            image = image.to(next(model.parameters()).device)
            generated_caption = model.generate_caption(image)

        # Compute ROUGE scores
        scores = rouge.get_scores(generated_caption, ground_truth)[0]
        rouge_scores["ROUGE-1"].append(scores["rouge-1"]["f"])
        rouge_scores["ROUGE-2"].append(scores["rouge-2"]["f"])
        rouge_scores["ROUGE-L"].append(scores["rouge-l"]["f"])

    # Average scores across all examples
    rouge_scores = {key: sum(values) / len(values) for key, values in rouge_scores.items()}
    return rouge_scores


def save_results(results, path):
    """
    Saves evaluation results to a JSON file.

    Args:
        results (dict): Evaluation results (e.g., BLEU and ROUGE scores).
        path (str): Path to save the results.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {path}")
