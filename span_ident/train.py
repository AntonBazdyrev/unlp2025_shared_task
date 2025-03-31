import click
import yaml
import wandb
import datetime
from data import load_datasets
from model import get_model
from trainer import get_trainer
from utils import set_seeds

@click.command()
@click.option('--val_fold', type=int, required=True, help='Validation fold index')
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to config.yaml')
def main(val_fold, config):
    # Load configuration
    with open(config, 'r') as f:
         config_data = yaml.safe_load(f)
        
    config_data['wandb']['run_name'] = config_data['wandb']['run_name'].format(
        val_fold=val_fold,
        date=datetime.date.today().strftime('%Y%m%d')
    )
    config_data["train_args"]["output_dir"] = config_data["train_args"]["output_dir"].format(run_name=config_data['wandb']['run_name'])
    config_data["train_args"]["logging_dir"] = config_data["train_args"]["logging_dir"].format(run_name=config_data['wandb']['run_name'])

    wandb.init(
        project=config_data["wandb"]["project"],
        entity=config_data["wandb"]["entity"],
        name=config_data['wandb']['run_name']
    )
    
    set_seeds(42)

    ds_train, ds_eval, ds_test, tokenizer, dataset_distribution = load_datasets(config_data, val_fold)
    
    model = get_model(config_data)

    trainer = get_trainer(model, config_data, ds_train, ds_eval, tokenizer, dataset_distribution)
    
    trainer.train()

if __name__ == '__main__':
    main()
