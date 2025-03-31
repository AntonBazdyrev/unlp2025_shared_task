import os
import click
import yaml
import datetime
import wandb
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

def extract_chars_from_spans(spans):
    """
    Given a list of spans (each a tuple (start, end)),
    return a set of character indices for all spans.
    """
    char_set = set()
    for start, end in spans:
        # Each span covers positions start, start+1, ..., end-1.
        char_set.update(range(start, end))
    return char_set
    
def inference_aggregation(probabilities, labels, offset_mappings, thold):
    predictions = (probabilities[:, :, 1] >= thold).astype(int)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)
    ]
    pred_spans_all = []
    for pred, offsets in zip(true_predictions, offset_mappings):
        samplewise_spans = []
        current_span = None
        for token_label, span in zip(pred, offsets):
            if token_label == 1:  # If the current token is labeled as an entity (1)
                if current_span is None:
                    current_span = [span[0], span[1]]  # Start a new span
                else:
                    current_span[1] = span[1]  # Extend the span to include the current token
            else:  # If token_label == 0 (not an entity)
                if current_span is not None:
                    samplewise_spans.append(tuple(current_span))  # Save completed span
                    current_span = None  # Reset for the next entity
        
                    # If the last token was part of a span, save it
        if current_span is not None:
            samplewise_spans.append(tuple(current_span))
        
        pred_spans_all.append(samplewise_spans)
    return [str(row) for row in pred_spans_all]


class THoptimizer:
    def __init__(self, ds):
        self.eval_dataset = ds
        
    def _calculate_inner_metric(self, gt_spans_all, pred_spans_all):
        total_true_chars = 0
        total_pred_chars = 0
        total_overlap_chars = 0
        for true_spans, pred_spans in zip(gt_spans_all, pred_spans_all):
            if isinstance(true_spans, str):
                try:
                    true_spans = eval(true_spans)
                except Exception:
                    true_spans = []
                    
            # Convert spans to sets of character indices.
            true_chars = extract_chars_from_spans(true_spans)
            pred_chars = extract_chars_from_spans(pred_spans)
            
            total_true_chars += len(true_chars)
            total_pred_chars += len(pred_chars)
            total_overlap_chars += len(true_chars.intersection(pred_chars))
            
            union_chars = true_chars.union(pred_chars)
            
        # Compute precision, recall, and F1.
        precision = total_overlap_chars / total_pred_chars if total_pred_chars > 0 else 0
        recall = total_overlap_chars / total_true_chars if total_true_chars > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        return metrics

    def _find_optimal_threshold(self, probabilities, labels):
        """Finds the threshold that achieves the desired positive class balance."""
        best_th = 0.5  # Default starting point
        best_diff = float("inf")
        optimal_th = best_th
        
        for thold in np.linspace(0.01, 0.99, num=100):
            predictions = (probabilities[:, :, 1] >= thold).astype(int)
            true_predictions = [
                [p for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            total_pos = sum([sum(row for row in prediction) for prediction in true_predictions])
            total = sum([len(prediction) for prediction in true_predictions])
            
            positive_ratio = total_pos / total if total > 0 else 0
            
            diff = abs(positive_ratio - self.desired_positive_ratio)
            if diff < best_diff:
                best_diff = diff
                optimal_th = thold
        
        return optimal_th
        
    def compute_metrics(self, eval_pred) -> dict:
        eval_dataset = self.eval_dataset
        logits, labels = eval_pred
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    
        thresholds = np.linspace(0.1, 0.5, num=41)
        #thresholds = [self._find_optimal_threshold(probabilities, labels)]
        results = []
        best_f1 = -1
        best_th = 0
        best_metrics = None
    
        for thold in tqdm(thresholds):
            # Apply thresholding instead of argmax
            predictions = (probabilities[:, :, 1] >= thold).astype(int)
    
            true_predictions = [
                [p for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
    
            pred_spans_all = []
            for pred, offsets in zip(true_predictions, eval_dataset['offset_mapping']):
                samplewise_spans = []
                current_span = None
                for token_label, span in zip(pred, offsets):
                    if token_label == 1:  # If the current token is labeled as an entity (1)
                        if current_span is None:
                            current_span = [span[0], span[1]]  # Start a new span
                        else:
                            current_span[1] = span[1]  # Extend the span to include the current token
                    else:  # If token_label == 0 (not an entity)
                        if current_span is not None:
                            samplewise_spans.append(tuple(current_span))  # Save completed span
                            current_span = None  # Reset for the next entity
    
                # If the last token was part of a span, save it
                if current_span is not None:
                    samplewise_spans.append(tuple(current_span))
    
                pred_spans_all.append(samplewise_spans)
    
            # Store results for this threshold
            current_metrics = self._calculate_inner_metric(eval_dataset['trigger_words'], pred_spans_all)
            if current_metrics['f1'] >= best_f1:
                best_f1 = current_metrics['f1']
                best_th = thold
                best_metrics = current_metrics
                best_metrics['thold'] = thold
                
            
            results.append(current_metrics)
        return best_metrics

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

    wandb.init(
        project=config_data["wandb"]["project"],
        entity=config_data["wandb"]["entity"],
        name=config_data['wandb']['run_name']
    )
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Load datasets (training, validation, test) and related info
    ds_train, ds_val, ds_test, tokenizer, dataset_distribution = load_datasets(config_data, val_fold)
    
    # Load the model and apply the LoRA weights from checkpoint
    model = get_model(config_data, load_lora=False)
    model = PeftModel.from_pretrained(model, checkpoint).eval()
    
    # Create trainer using your custom trainer class
    trainer = get_trainer(
        model, config_data, ds_train, ds_val, tokenizer, dataset_distribution, do_train=False
    )
    
    # -------------------------------
    # Inference on validation and test sets
    # -------------------------------
    click.echo("Running inference on validation set...")
    val_output = trainer.predict(ds_val)
    val_logits = val_output.predictions  # shape: (num_samples, num_classes)
    val_labels = val_output.label_ids    # ground-truth labels


    val_metrics = trainer.compute_metrics((val_logits, val_labels))
    val_metrics = {k: float(v) for k, v in val_metrics.items()}

    click.echo(f"val metrics class balance: {val_metrics}")

    thoptimizer = THoptimizer(trainer.eval_dataset)
    val_metrics_gs = thoptimizer.compute_metrics((val_logits, val_labels))
    val_metrics_gs = {k: float(v) for k, v in val_metrics_gs.items()}

    click.echo(f"val metrics grid search: {val_metrics_gs}")
    
    
    click.echo("Running inference on test set...")
    test_output = trainer.predict(ds_test)
    test_logits = test_output.predictions  # shape: (num_samples, num_classes)
    test_labels = test_output.label_ids     #mock-up labels with padding
    
    # Convert logits to probabilities using sigmoid

    valid_probabilities = torch.softmax(torch.tensor(val_logits), dim=-1).cpu().numpy()
    test_probabilities = torch.softmax(torch.tensor(test_logits), dim=-1).cpu().numpy()

    np.save(os.path.join(result_dir, "val_probs.npy"), valid_probabilities)
    np.save(os.path.join(result_dir, "test_probs.npy"), test_probabilities)

    np.save(os.path.join(result_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(result_dir, "test_labels.npy"), test_labels)

    with open(os.path.join(result_dir, "thresholds_class_balance.yaml"), "w") as f:
        yaml.dump(val_metrics, f)
        
    with open(os.path.join(result_dir, "thresholds_grid_search.yaml"), "w") as f:
        yaml.dump(val_metrics_gs, f)
    
    click.echo("Inference complete. Probabilities, thresholds, and submissions have been saved to the result directory.")

if __name__ == '__main__':
    main()
