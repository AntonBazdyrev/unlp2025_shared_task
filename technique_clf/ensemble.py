import os
import yaml
import numpy as np
import click
import pandas as pd

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

    # Load configuration to get the sample submission path.
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
    sample_submission_path = os.path.join(config_data["dataset"]["data_dir"],
                                          config_data["dataset"]["sample_submission_path"])

    test_probs_list = []
    thresholds_list = []
    
    # Determine the key in the thresholds.yaml based on chosen threshold type.
    th_key = {
        'grid': 'grid_thresholds',
        'distribution': 'distribution_thresholds',
        'regularized': 'regularized_thresholds'
    }[th_type]
    
    click.echo("Loading probabilities and thresholds from each fold...")
    for fold_dir in fold_dirs:
        probs_path = os.path.join(fold_dir, "test_probs.npy")
        th_path = os.path.join(fold_dir, "thresholds.yaml")
        
        if not os.path.exists(probs_path):
            click.echo(f"File not found: {probs_path}")
            continue
        if not os.path.exists(th_path):
            click.echo(f"File not found: {th_path}")
            continue
        
        # Load test probabilities
        probs = np.load(probs_path)
        test_probs_list.append(probs)
        
        # Load thresholds
        with open(th_path, "r") as f:
            th_data = yaml.load(f, Loader=yaml.UnsafeLoader)
        # Expecting the thresholds to be stored under the appropriate key
        if th_key not in th_data:
            click.echo(f"Threshold key '{th_key}' not found in {th_path}.")
            continue
        thresholds = th_data[th_key]
        thresholds_list.append(np.array(thresholds))
    
    if not test_probs_list or not thresholds_list:
        click.echo("No valid data found. Exiting.")
        return

    # --- Ensemble Average ---
    # Average probabilities elementwise
    ensemble_probs = np.mean(np.array(test_probs_list), axis=0)
    # Average thresholds for each class
    ensemble_thresholds = np.mean(np.array(thresholds_list), axis=0)
    
    click.echo("Averaged test probabilities and thresholds computed.")

    # --- Binarize ensemble probabilities ---
    ensemble_preds = (ensemble_probs >= ensemble_thresholds).astype(int)
    
    # --- Save Ensemble Outputs ---
    probs_save_path = os.path.join(output_dir, "ensemble_test_probs.npy")
    np.save(probs_save_path, ensemble_probs)
    
    th_save_dict = {
        th_key: ensemble_thresholds.tolist()
    }
    thresholds_save_path = os.path.join(output_dir, "ensemble_thresholds.yaml")
    with open(thresholds_save_path, "w") as f:
        yaml.dump(th_save_dict, f)
    
    click.echo(f"Ensemble test probabilities saved to: {probs_save_path}")
    click.echo(f"Ensemble thresholds saved to: {thresholds_save_path}")

    # --- Create Ensemble Submission ---
    submission = pd.read_csv(sample_submission_path)
    # Assume the submission file has an 'id' column and remaining columns for each target label.
    label_cols = [col for col in submission.columns if col != "id"]
    if ensemble_preds.shape[1] != len(label_cols):
        click.echo("Mismatch between number of prediction columns and submission labels.")
        return
    
    # Fill predictions into the submission DataFrame
    for i, col in enumerate(label_cols):
        submission[col] = ensemble_preds[:, i]
    
    submission_filename = f"ensemble_{th_type}.csv"
    submission_path = os.path.join(output_dir, submission_filename)
    submission.to_csv(submission_path, index=False)
    
    click.echo(f"Ensemble submission saved to: {submission_path}")

if __name__ == '__main__':
    main()
