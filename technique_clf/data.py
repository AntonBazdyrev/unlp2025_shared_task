import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from collections.abc import Iterable

def load_datasets(config, val_fold):
    data_dir = config['dataset']['data_dir']
    train_path = os.path.join(data_dir, config['dataset']['train_path'])
    cv_path = os.path.join(data_dir, config['dataset']['cv_path'])
    test_path = os.path.join(data_dir, config['dataset']['test_path'])
    sample_submission_path = os.path.join(data_dir, config['dataset']['sample_submission_path'])

    # Load and merge training data
    df = pd.read_parquet(train_path)
    cv = pd.read_csv(cv_path)
    df = df.merge(cv, on='id', how='left')
    test = pd.read_csv(test_path)

    # Set validation split based on the provided fold index
    df['is_valid'] = df.fold == val_fold

    # Create prompts using the instruction from config
    instruction = config['prompt_generator']['instruction']
    def prompt_generator(text):
        return f"""<start_of_turn>user
{instruction}
Текст статті: {text}
<end_of_turn>"""
    df['prompt'] = df['content'].apply(prompt_generator)
    test['prompt'] = test['content'].apply(prompt_generator)

    # Initialize tokenizer
    pretrained_model = config['model']['pretrained_model']
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.padding_side = 'right'
    tokenizer.add_eos_token = True
    max_length = config['model']['max_length']

    # Process full text
    df['full_text'] = df['prompt'].apply(
        lambda x: tokenizer.decode(tokenizer(x, add_special_tokens=False)['input_ids'][:max_length])
    )
    test['full_text'] = test['prompt'].apply(
        lambda x: tokenizer.decode(tokenizer(x, add_special_tokens=False)['input_ids'][:max_length])
    )

    # Prepare labels based on sample submission columns
    ssubmission = pd.read_csv(sample_submission_path)
    targets = ssubmission.set_index('id').columns.tolist()

    for col in targets:
        df[col] = 0
    
    for ind, row in df.iterrows():
        if isinstance(row['techniques'], Iterable):
            for t in row['techniques']:
                df.loc[ind, t] = 1
        df['labels'] = list(df[targets].values)


    def tokenize(sample):
        return tokenizer(sample['full_text'])

        
    df['labels'] = list(df[targets].values)
    ds_train = Dataset.from_pandas(df[df.is_valid == 0][['full_text', 'labels']].copy())
    ds_eval = Dataset.from_pandas(df[df.is_valid == 1][['full_text', 'labels']].copy())
    ds_test = Dataset.from_pandas(test[['full_text']].copy())
    
    ds_train = ds_train.map(tokenize)
    remove_columns = [c for c in ds_train.features.keys() if c not in ['input_ids', 'attention_mask', 'labels']]
    ds_train = ds_train.remove_columns(remove_columns)
    
    ds_eval = ds_eval.map(tokenize)
    remove_columns = [c for c in ds_eval.features.keys() if c not in ['input_ids', 'attention_mask', 'labels']]
    ds_eval = ds_eval.remove_columns(remove_columns)
    
    ds_test = ds_test.map(tokenize)
    remove_columns = [c for c in ds_test.features.keys() if c not in ['input_ids', 'attention_mask', 'labels']]
    ds_test = ds_test.remove_columns(remove_columns)

    dataset_distribution = df[targets].mean().values

    print(dataset_distribution)

    return ds_train, ds_eval, ds_test, targets, tokenizer, dataset_distribution
