import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from collections.abc import Iterable
from itertools import chain

import spacy

from spacy.training.iob_utils import biluo_to_iob, doc_to_biluo_tags
from tqdm import tqdm


def convert_to_seq_labeling(text, tokenizer, trigger_spans=None, MAX_LEN=None):
    tokenized_output = tokenizer(
        text, return_offsets_mapping=True, add_special_tokens=True, max_length=MAX_LEN,
        truncation=True, padding=False
    )
    tokens = tokenized_output["input_ids"]
    offsets = tokenized_output["offset_mapping"]

    # Get subword tokenized versions of the text
    token_strings = tokenizer.convert_ids_to_tokens(tokens)

    
    # Initialize labels as 'O'
    labels = [0] * len(tokens)

    if trigger_spans is not None:
        # Assign 'TRIGGER' to overlapping tokens
        for start, end in trigger_spans:
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == 0 and tok_end == 0:
                    continue
                if tok_start < end and tok_end > start:  # If token overlaps with the trigger span
                    labels[i] = 1

    tokenized_output['labels'] = labels
    return tokenized_output


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
    df_test = pd.read_csv(test_path)

    # Set validation split based on the provided fold index
    df['is_valid'] = df.fold == val_fold
    print(val_fold)
    print(df.is_valid.value_counts())

    MAX_LEN = config['model']['max_length']
    PRETRAINED_MODEL = config['model']['pretrained_model']
    df.trigger_words = df.trigger_words.apply(lambda x: [] if x is None else x)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    df['seq_labels'] = df.apply(lambda row: convert_to_seq_labeling(row['content'], tokenizer, row['trigger_words'], MAX_LEN), axis=1)
    for column in df.seq_labels.iloc[0].keys():
        df[column] = df.seq_labels.apply(lambda x: x.get(column))
    
    df_test['seq_labels'] = df_test.apply(lambda row: convert_to_seq_labeling(row['content'], tokenizer, None, MAX_LEN), axis=1)
    for column in df_test.seq_labels.iloc[0].keys():
        df_test[column] = df_test.seq_labels.apply(lambda x: x.get(column))

    columns = list(df.seq_labels.iloc[0].keys()) + ['content', 'trigger_words']
    ds_train = Dataset.from_pandas(df[df.is_valid==0][columns].reset_index(drop=True))
    ds_valid = Dataset.from_pandas(df[df.is_valid==1][columns].reset_index(drop=True))
    
    columns = list(df.seq_labels.iloc[0].keys()) + ['content']
    ds_test = Dataset.from_pandas(df_test[columns].reset_index(drop=True))

    positive_class_balance = pd.Series(list(chain(*df.labels.tolist()))).mean()
    
    return ds_train, ds_valid, ds_test, tokenizer, positive_class_balance
