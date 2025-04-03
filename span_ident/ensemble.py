import os
import yaml
import numpy as np
import click
import pandas as pd
import datetime
import wandb
from tqdm import tqdm

# Import helper functions and modules from your project
from data import load_datasets
from model import get_model
from trainer import get_trainer
from utils import set_seeds

# --- YAML Constructors for NumPy Objects ---
def numpy_reconstruct_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    args = mapping.get('args', [])
    state = mapping.get('state', None)
    arr = np.core.multiarray._reconstruct(*args)
    if state is not None:
        arr.__setstate__(state)
    return arr

def numpy_scalar_constructor(loader, node):
    args = loader.construct_sequence(node, deep=True)
    return np.core.multiarray.scalar(*args)

yaml.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray._reconstruct',
    numpy_reconstruct_constructor,
    Loader=yaml.UnsafeLoader
)
yaml.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
    numpy_scalar_constructor,
    Loader=yaml.UnsafeLoader
)

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

# --- Ensemble Script ---
@click.command()
@click.option('--fold_dirs', multiple=True, required=True, help="Directories for each fold (each must contain test_probs.npy and thresholds.yaml)")
@click.option('--config', type=click.Path(exists=True), required=True, help="Path to config.yaml")
@click.option('--output_dir', type=click.Path(), required=True, help="Directory to save ensemble results")
@click.option('--th_type', type=click.Choice(['grid', 'distribution', 'regularized']), default='regularized', help="Threshold type to use for ensemble")
def main(fold_dirs, config, output_dir, th_type):
    """
    Ensemble prediction by averaging test probabilities and thresholds.
    Loads test probabilities and thresholds from each fold directory,
    averages them (equal weighting), then binarizes the ensemble probabilities
    using the averaged thresholds. The sample submission path is taken from the config.yaml.
    The ensemble submission file is saved as: ensemble_{th_type}.csv in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration and update paths/run names
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Load datasets (training, validation, test) and related info
    ds_train, ds_val, ds_test, tokenizer, dataset_distribution = load_datasets(config_data, -1)

    # Load configuration to get the sample submission path.
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
    sample_submission_path = os.path.join(config_data["dataset"]["data_dir"],
                                          config_data["dataset"]["sample_submission_path"])
    submission = pd.read_csv(sample_submission_path)

    tholds_gs = []
    tholds_cb = []
    test_probs_list = []
    test_mask_list = []
    for fold_dir in fold_dirs:
        with open(os.path.join(fold_dir, "thresholds_grid_search.yaml"), "r") as f:
            th_data_gs = yaml.load(f, Loader=yaml.UnsafeLoader)
        tholds_gs.append(th_data_gs['thold'])
            
        with open(os.path.join(fold_dir, "thresholds_class_balance.yaml"), "r") as f:
            th_data_cb = yaml.load(f, Loader=yaml.UnsafeLoader)
        tholds_cb.append(th_data_cb['thold'])

        probs_path = os.path.join(fold_dir, "test_probs.npy")
        probs = np.load(probs_path)
        test_probs_list.append(probs)

        mask_path = os.path.join(fold_dir, "test_labels.npy")
        mask = np.load(mask_path)
        test_mask_list.append(mask)

    
    tholds_gs = np.array(tholds_gs)
    tholds_cb = np.array(tholds_cb)

    mask_labels = test_mask_list[0]

    ensemble_probs = np.mean(np.array(test_probs_list), axis=0)
    # Average thresholds for each class
    ensemble_thresholds_gs = np.mean(np.array(tholds_gs))
    ensemble_thresholds_cb = np.mean(np.array(tholds_cb))
    ensemble_thresholds_regularized = 0.5*ensemble_thresholds_gs + 0.5*ensemble_thresholds_cb

    print(ensemble_thresholds_gs, ensemble_thresholds_regularized)
    test_results_gs = inference_aggregation(
        ensemble_probs, mask_labels, ds_test['offset_mapping'], ensemble_thresholds_gs)
    test_results_regularized = inference_aggregation(
        ensemble_probs, mask_labels, ds_test['offset_mapping'], ensemble_thresholds_regularized)

    submission['trigger_words'] = test_results_gs
    submission_filename = f"ensemble_gs.csv"
    submission_path = os.path.join(output_dir, submission_filename)
    submission.to_csv(submission_path, index=False)

    submission['trigger_words'] = test_results_regularized
    submission_filename = f"ensemble_reg.csv"
    submission_path = os.path.join(output_dir, submission_filename)
    submission.to_csv(submission_path, index=False)

    np.save(os.path.join(output_dir, "ensemble_probs.npy"), ensemble_probs)
    np.save(os.path.join(output_dir, "ensemble_mask.npy"), mask_labels)
    
    with open(os.path.join(output_dir, "thresholds.yaml"), "w") as f:
        yaml.dump({'gs': float(ensemble_thresholds_gs), 'reg': float(ensemble_thresholds_regularized)}, f)
   
    click.echo(f"Ensemble submission saved to: {submission_path}")

if __name__ == '__main__':
    main()
