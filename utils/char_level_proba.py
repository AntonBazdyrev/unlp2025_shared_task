import pandas as pd

def inference_to_char_level(probabilities, labels, offset_mappings, ids):
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(probabilities[:, :, 1], labels)
    ]
    
    all_data = []
    
    for sample_id, pred, offsets, probs in zip(ids, true_predictions, offset_mappings, probabilities[:, :, 1]):
        for token_label, span, proba in zip(pred, offsets, probs):
            for char_index in range(span[0], span[1]):
                all_data.append({"id": sample_id, "char_index": char_index, "proba": proba})

    output_df = pd.DataFrame(all_data)

    # something was wrong, got some duplicates, mb because of out of vocab tokens
    output_df = output_df.groupby(['id', 'char_index'], as_index=False)['proba'].mean()

    return output_df


def char_level_to_spans(df, thold):
    spans_all = []
    for sample_id, group in df.groupby("id"):
        spans = []
        current_span = None
        for _, row in group.iterrows():
            if row["proba"] >= thold:
                if current_span is None:
                    current_span = [row["char_index"], row["char_index"] + 1]
                else:
                    current_span[1] = row["char_index"] + 1
            else:
                if current_span is not None:
                    spans.append(tuple(current_span))
                    current_span = None
        if current_span is not None:
            spans.append(tuple(current_span))
        spans_all.append({"id": sample_id, "trigger_words": spans})
    return pd.DataFrame(spans_all)