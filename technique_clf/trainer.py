import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import f1_score
import numpy as np
from scipy.optimize import minimize_scalar

def find_thresholds_for_distribution(preds, desired_distribution):
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

def compute_scaled_class_weights_from_proportions(dataset_distribution, exponent=0.5, clip_value=None, norm_type='sum'):
    distribution = np.array(dataset_distribution, dtype=np.float32)
    
    # Compute raw weights using reciprocal since distribution is already a fraction.
    raw_weights = 1.0 / distribution  # rarer classes (low proportion) yield higher values.
    
    # Apply power transform to dampen the extremes.
    scaled_weights = raw_weights ** exponent
    
    # Optionally clip weights to avoid extremely high values.
    if clip_value is not None:
        scaled_weights = np.minimum(scaled_weights, clip_value)
    
    # Normalize the weights if needed.
    if norm_type == 'sum':
        # Normalize so that the sum equals the number of classes.
        scaled_weights = scaled_weights / scaled_weights.sum() * len(scaled_weights)
    
    return scaled_weights

def compute_metrics(eval_pred, target_distribution):
    logits, labels = eval_pred
    proba = torch.nn.functional.sigmoid(torch.tensor(logits)).numpy()
    optimal_thresholds = find_thresholds_for_distribution(proba, desired_distribution=target_distribution)
    binarized_preds = (proba >= np.array(optimal_thresholds)).astype(int)
    return {"f1": f1_score(labels, binarized_preds, average="macro")}

class CustomTrainer(Trainer):
    def __init__(self, *args, targets=None, dataset_distribution=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = targets
        #self.class_weights = None
        scaled_weights = compute_scaled_class_weights_from_proportions(
            dataset_distribution, exponent=0.5, clip_value=5.0
        )
        print("Scaled weights:", scaled_weights)
        self.class_weights = torch.tensor(scaled_weights).cuda()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(weight=self.class_weights) if self.class_weights is not None else torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        logits = outputs.logits
        loss = self.loss_fn(logits, inputs['labels'].float())
        return (loss, outputs) if return_outputs else loss

def get_trainer(model, config, ds_train, ds_eval, targets, tokenizer, dataset_distribution, do_train=True):
    training_args = TrainingArguments(
        **config['train_args']
    )
    if not do_train:
        training_args.report_to='none'
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, dataset_distribution),
        targets=targets,
        dataset_distribution=dataset_distribution
    )
    return trainer
