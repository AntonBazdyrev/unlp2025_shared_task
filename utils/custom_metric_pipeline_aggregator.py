import math
import numpy as np
import warnings
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from enum import Enum
from typing import Any, List, Tuple, Optional

# --- Replicate the AggregationStrategy Enum from the official code ---
class AggregationStrategy(Enum):
    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"

def extract_chars_from_spans(spans: List[Tuple[int, int]]) -> set:
    """
    Given a list of spans (each a tuple (start, end)),
    return a set of character indices covered by all spans.
    """
    char_set = set()
    for start, end in spans:
        char_set.update(range(start, end))
    return char_set

class SpanEvaluationTrainer(Trainer):
    def __init__(
        self,
        model: Any = None,
        args: TrainingArguments = None,
        data_collator: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        **kwargs,
    ):
        """
        Initialize the Trainer with our custom compute_metrics.
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,  # use our custom compute_metrics
            **kwargs,
        )
        # For simplicity, assume PyTorch.
        self.framework = "pt"

    # --- Aggregation functions borrowed from the official pipeline ---

    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: List[int],
        scores: np.ndarray,
        offset_mapping: List[Tuple[int, int]],
        special_tokens_mask: List[int],
        aggregation_strategy: AggregationStrategy,
    ) -> List[dict]:
        """
        Fuse various numpy arrays into dicts with all the information needed for aggregation.
        This method is a nearly identical copy of the official pipeline's gather_pre_entities.
        """
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Skip special tokens.
            if special_tokens_mask[idx]:
                continue
            word = self.processing_class.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                if not isinstance(start_ind, int):
                    if self.framework == "pt":
                        start_ind = int(start_ind.item())
                        end_ind = int(end_ind.item())
                word_ref = sentence[start_ind:end_ind]
                # If using a fast tokenizer with BPE, check for subwords.
                if getattr(self.processing_class, "_tokenizer", None) and getattr(self.processing_class._tokenizer.model, "continuing_subword_prefix", None):
                    is_subword = len(word) != len(word_ref)
                else:
                    # Fallback heuristic.
                    if aggregation_strategy in {AggregationStrategy.FIRST, AggregationStrategy.AVERAGE, AggregationStrategy.MAX}:
                        warnings.warn("Tokenizer does not support real words, using fallback heuristic", UserWarning)
                    is_subword = start_ind > 0 and " " not in sentence[start_ind - 1 : start_ind + 1]
                if int(input_ids[idx]) == self.processing_class.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind, end_ind = None, None
                is_subword = False
            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        """
        For each token, choose its highest-scoring label, then (if not NONE) group adjacent tokens.
        This mimics the "simple" aggregation strategy.
        """
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.model.config.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            # (Other strategies not implemented here.)
            entities = []
        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Group together adjacent tokens with overlapping spans.
        This is a simplified copy of the official pipeline's grouping method.
        """
        if len(entities) == 0:
            return entities
        entities = sorted(entities, key=lambda x: x["start"] if x["start"] is not None else 0)
        aggregated_entities = []
        previous_entity = entities[0]
        for entity in entities[1:]:
            if previous_entity["start"] <= entity["start"] < previous_entity["end"]:
                current_length = entity["end"] - entity["start"]
                previous_length = previous_entity["end"] - previous_entity["start"]
                if current_length > previous_length:
                    previous_entity = entity
                elif current_length == previous_length and entity["score"] > previous_entity["score"]:
                    previous_entity = entity
            else:
                aggregated_entities.append(previous_entity)
                previous_entity = entity
        aggregated_entities.append(previous_entity)
        return aggregated_entities

    def aggregate_logits(self, sample_logits: np.ndarray, text: str, seq_length: int) -> List[Tuple[int, int]]:
        """
        Given the logits for a sample and its text, re-tokenize the text with padding set to seq_length
        and then use the borrowed aggregator functions to produce predicted spans.
        """
        tokenized = self.processing_class(
            text,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=True,
            padding="max_length",
            max_length=seq_length,
        )
        tokenized["sentence"] = text

        print('LEN OF TOKENS = ', len(tokenized["input_ids"]))
        print('LEN OF LOGITS = ', len(sample_logits))
        
        pre_entities = self.gather_pre_entities(
            tokenized["sentence"],
            tokenized["input_ids"],
            sample_logits,
            tokenized["offset_mapping"],
            tokenized["special_tokens_mask"],
            AggregationStrategy.SIMPLE,
        )
        aggregated = self.aggregate(pre_entities, AggregationStrategy.SIMPLE)
        if AggregationStrategy.SIMPLE != AggregationStrategy.NONE:
            aggregated = self.group_entities(aggregated)
        # Filter out "O" labels.
        entities = [
            entity
            for entity in aggregated
            if entity.get("entity") != "O" and entity.get("entity_group", None) != "O"
        ]
        spans = [
            (entity["start"], entity["end"])
            for entity in entities
            if entity["start"] is not None and entity["end"] is not None
        ]
        return spans

    # --- End of Aggregation Functions ---

    def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        """
        Compute character-level span metrics (precision, recall, F1, accuracy) using the logits produced by evaluation.
        
        Assumptions:
          - eval_pred.predictions is a NumPy array of shape (num_samples, seq_length, num_labels).
          - self.eval_dataset is an iterable of dicts, each containing:
                "content": the original text,
                "trigger_words": the ground truth spans (list of tuples or string representation).
          - We re-tokenize each text (forcing the same seq_length as used during evaluation) and then use our
            aggregation functions to produce predicted spans.
        """
        predictions = eval_pred.predictions  # shape: (N, L, num_labels)
        eval_dataset = self.eval_dataset

        total_true_chars = 0
        total_pred_chars = 0
        total_overlap_chars = 0
        total_chars = 0
        total_correct_chars = 0

        for i, sample in enumerate(eval_dataset):
            text = sample["content"]
            L_text = len(text)
            total_chars += L_text

            # Get ground truth spans.
            true_spans = sample["trigger_words"]
            if isinstance(true_spans, str):
                try:
                    true_spans = eval(true_spans)
                except Exception:
                    true_spans = []

            # Get logits for this sample.
            sample_logits = predictions[i]  # shape (seq_length, num_labels)
            seq_length = sample_logits.shape[0]

            # Use the aggregator to produce predicted spans.
            pred_spans = self.aggregate_logits(sample_logits, text, seq_length)

            # Expand both ground truth and predicted spans to sets of character indices.
            true_chars = extract_chars_from_spans(true_spans)
            pred_chars = extract_chars_from_spans(pred_spans)

            total_true_chars += len(true_chars)
            total_pred_chars += len(pred_chars)
            total_overlap_chars += len(true_chars.intersection(pred_chars))

            union_chars = true_chars.union(pred_chars)
            correct_chars = len(true_chars.intersection(pred_chars)) + (L_text - len(union_chars))
            total_correct_chars += correct_chars

        precision = total_overlap_chars / total_pred_chars if total_pred_chars > 0 else 0
        recall = total_overlap_chars / total_true_chars if total_true_chars > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = total_correct_chars / total_chars if total_chars > 0 else 0

        metrics = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
        return metrics
    