import os
import click
import yaml
import datetime
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

# Import helper functions and modules from your project
from data import load_datasets
from model import get_model
from trainer import get_trainer
from utils import set_seeds
from peft import PeftModel

# -------------------------------
# Threshold optimization functions
# -------------------------------

def optimize_thresholds(preds, targets, num_thresholds=100):
    """
    Find optimal thresholds for each class to maximize average F1 score.
    
    Args:
        preds (ndarray): Array of shape (num_samples, num_classes) with probabilities.
        targets (ndarray): Array of shape (num_samples, num_classes) with binary ground-truth labels.
        num_thresholds (int): Number of thresholds to evaluate (default: 100).

    Returns:
        optimal_thresholds (list): List of optimal thresholds for each class.
        best_avg_f1 (float): Best average F1 score achieved.
    """
    num_classes = preds.shape[1]
    thresholds = np.linspace(0, 1, num_thresholds)
    optimal_thresholds = []
    best_avg_f1 = 0

    for class_idx in tqdm(range(num_classes), desc="Grid search per class"):
        best_f1 = 0
        best_threshold = 0
        for threshold in thresholds:
            # Binarize predictions for this class
            binarized_preds = (preds[:, class_idx] >= threshold).astype(int)
            f1 = f1_score(targets[:, class_idx], binarized_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        optimal_thresholds.append(best_threshold)
    
    # Compute average F1 across classes using optimal thresholds
    binarized_preds = (preds >= np.array(optimal_thresholds)).astype(int)
    avg_f1 = f1_score(targets, binarized_preds, average='macro', zero_division=0)

    return optimal_thresholds, avg_f1

def find_thresholds_for_distribution(preds, desired_distribution):
    """
    Find thresholds for each class to achieve the desired class distribution.

    Args:
        preds (ndarray): Array of shape (num_samples, num_classes) with probabilities.
        desired_distribution (list): Desired proportion of positive samples for each class.

    Returns:
        thresholds (list): List of thresholds for each class.
    """
    num_classes = preds.shape[1]
    thresholds = []

    for class_idx in range(num_classes):
        probs = preds[:, class_idx]
        desired_ratio = desired_distribution[class_idx]

        def objective(threshold):
            predicted_ratio = (probs >= threshold).mean()
            return abs(predicted_ratio - desired_ratio)

        result = minimize_scalar(objective, bounds=(0, 1), method="bounded")
        thresholds.append(result.x)
    return thresholds

# -------------------------------
# Inference script entrypoint
# -------------------------------

@click.command()
@click.option('--val_fold', type=int, required=True, help='Validation fold index')
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to config.yaml')
@click.option('--checkpoint', type=click.Path(exists=True), required=True, help='Path to model checkpoint (LoRA weights)')
@click.option('--result_dir', type=click.Path(), required=True, help='Directory to save results (submissions, thresholds, and probs)')
def main(val_fold, config, checkpoint, result_dir):
    # Create the result directory if it does not exist
    os.makedirs(result_dir, exist_ok=True)

    # Load configuration and update paths/run names
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    config_data['wandb']['run_name'] = config_data['wandb']['run_name'].format(
        val_fold=val_fold,
        date=datetime.date.today().strftime('%Y%m%d')
    )
    config_data["train_args"]["output_dir"] = config_data["train_args"]["output_dir"].format(
        run_name=config_data['wandb']['run_name']
    )
    config_data["train_args"]["logging_dir"] = config_data["train_args"]["logging_dir"].format(
        run_name=config_data['wandb']['run_name']
    )
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Load datasets (training, validation, test) and related info
    ds_train, ds_val, ds_test, targets, tokenizer, dataset_distribution = load_datasets(config_data, val_fold)
    
    # Load the model and apply the LoRA weights from checkpoint
    model = get_model(config_data, num_labels=len(targets), load_lora=False)
    model = PeftModel.from_pretrained(model, checkpoint).eval()
    
    # Create trainer using your custom trainer class
    trainer = get_trainer(
        model, config_data, ds_train, ds_val, targets, tokenizer, dataset_distribution, do_train=False
    )
    
    # -------------------------------
    # Inference on validation and test sets
    # -------------------------------
    click.echo("Running inference on validation set...")
    val_output = trainer.predict(ds_val)
    val_logits = val_output.predictions  # shape: (num_samples, num_classes)
    val_labels = val_output.label_ids    # ground-truth labels
    
    click.echo("Running inference on test set...")
    test_output = trainer.predict(ds_test)
    test_logits = test_output.predictions  # shape: (num_samples, num_classes)
    
    # Convert logits to probabilities using sigmoid
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    val_probs = sigmoid(val_logits)
    test_probs = sigmoid(test_logits)
    
    # -------------------------------
    # Threshold computation
    # -------------------------------
    click.echo("Optimizing thresholds using grid search on validation set...")
    grid_thresholds, grid_avg_f1 = optimize_thresholds(val_probs, val_labels, num_thresholds=100)
    
    click.echo("Calculating distribution-based thresholds on test set predictions...")
    dist_thresholds = find_thresholds_for_distribution(test_probs, desired_distribution=dataset_distribution)

    grid_thresholds = np.array(grid_thresholds)
    dist_thresholds = np.array(dist_thresholds)

    click.echo(f"grids tholds: {grid_thresholds}")
    click.echo(f"distr tholds: {dist_thresholds}")

    reg_thresholds = 0.5*grid_thresholds + 0.5*dist_thresholds

    click.echo(f"reg tholds: {dist_thresholds}")
    
    # -------------------------------
    # Evaluation on validation set
    # -------------------------------
    def compute_val_f1(probs, labels, thresholds):
        binarized = (probs >= np.array(thresholds)).astype(int)
        return f1_score(labels, binarized, average='macro', zero_division=0)
    
    grid_val_f1 = compute_val_f1(val_probs, val_labels, grid_thresholds)
    dist_val_f1 = compute_val_f1(val_probs, val_labels, dist_thresholds)
    reg_val_f1 = compute_val_f1(val_probs, val_labels, reg_thresholds)
    
    click.echo("Validation F1 Scores:")
    click.echo(f"  Grid Search Thresholds F1: {grid_val_f1:.4f}")
    click.echo(f"  Distribution-based Thresholds F1: {dist_val_f1:.4f}")
    click.echo(f"  Regularized Thresholds F1: {reg_val_f1:.4f}")
    
    # -------------------------------
    # Save probabilities and thresholds
    # -------------------------------
    np.save(os.path.join(result_dir, "val_probs.npy"), val_probs)
    np.save(os.path.join(result_dir, "test_probs.npy"), test_probs)
    
    thresholds_dict = {
        "grid_thresholds": grid_thresholds,
        "distribution_thresholds": dist_thresholds,
        "regularized_thresholds": reg_thresholds,
        "grid_val_f1": grid_val_f1,
        "distribution_val_f1": dist_val_f1,
        "regularized_val_f1": reg_val_f1
    }
    with open(os.path.join(result_dir, "thresholds.yaml"), "w") as f:
        yaml.dump(thresholds_dict, f)
    
    # -------------------------------
    # Create and save submissions
    # -------------------------------
    # Load sample submission template
    sample_submission_path = os.path.join(config_data['dataset']['data_dir'], config_data['dataset']['sample_submission_path'])
    submission_df = pd.read_csv(sample_submission_path)
    
    def create_submission(probs, thresholds, th_type, val_f1):
        preds = (probs >= np.array(thresholds)).astype(int)
        submission = submission_df.copy()
        # Assume submission file has an 'id' column and one column per target label
        for i, col in enumerate(targets):
            submission[col] = preds[:, i]
        filename = os.path.join(result_dir, f"submission_{val_fold}_{th_type}_{val_f1:.4f}.csv")
        submission.to_csv(filename, index=False)
        click.echo(f"Saved submission: {filename}")
    
    create_submission(test_probs, grid_thresholds, "grid", grid_val_f1)
    create_submission(test_probs, dist_thresholds, "distribution", dist_val_f1)
    create_submission(test_probs, reg_thresholds, "regularized", reg_val_f1)
    
    click.echo("Inference complete. Probabilities, thresholds, and submissions have been saved to the result directory.")

if __name__ == '__main__':
    main()
