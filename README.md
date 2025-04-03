# unlp2025_shared_task

# Technique Classification 

**Overview**

Our solution is out of fold ensemble prediction of the Gemma2-27b (model for sequence classification) for 5 folds with custom threshold tuning specifically for F1 metric optimization. 

You can check the source code in the [repo](https://github.com/AntonBazdyrev/unlp2025_shared_task/tree/master)

**LB probing**

At the very beginning, we did an LB probing to check the actual class distribution for the test dataset (public). We needed to do so because the competition metric is macro F1, which is highly dependent on the quality of prediction for all classes and, in practice, on minority classes, which are hard to predict.

You can check, that in the binary case for "constant all 1" prediction knowing the F1 score, you can calculate the positive class balance
$$ balance = \frac{F1}{2-F1}$$
In our case, where we have a multilabel task with a macro F1 score, it's equivalent to the mean marginal F1 scores for all 10 classes. So, if we submit "constant all 1" predictions separately for each class with "constant all 0" for other classes - we can get marginal F1 scores for all classes and calculate the actual class balances for the test set.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1697396%2F7ae68558dfdc567000ea2d3f6a299d49%2FScreenshot%202025-04-03%20at%2020.19.19.png?generation=1743701138110689&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1697396%2F0f24c93912ce4d5425fdc64a96328770%2FScreenshot%202025-04-03%20at%2020.19.42.png?generation=1743701146145406&alt=media)

**Validation**
According to the LB probing section - we can check that it seems that the dataset split is something like a multi-label stratified shuffle split, so we used:
```
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mlskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
According to our experiments, we had a perfect CV-LB correlation. For some experiments we used a 5-fold CV, but mostly, we used a single-fold holdout. Because the LB split is shared for both tasks - we used the same split for the span identification task too.

**Thresholds Optimization**
Usually, for tasks with a huge class imbalance, the default 0.5 th is not optimal at all.
Because thresholds are independent in this case, we can do a grid search optimization for each class separately. The optimal that maximizes the F1 score on validation:

$$t_{gs}^* = \operatorname{arg\,max}_{t \in [0,1]} F1\_{val}(t)$$

With this strategy we can achieve the best possible F1 on the local CV, but because of the its nature - it results in overfitting on the LB.

So, there is another more robust option for threshold selection with respect to the class balance. Important note, you can check that this approach produces the near-optimal threshold for the near-perfect estimator.

$$t_{cb}^* = \operatorname{arg\,min}_{t \in [0,1]} |r(t) - r^*|$$

Where, r* is the true positive class ratio and r(t) is the positive class ratio in the prediction with threshold t. This class balance based threshold results in a more stable performance, but in terms of the absolute values it produces suboptimal result.

So, we can merge these approaches into a hybrid grid search thresholds regularized with respect to the class balance where:

$$t_{final}^* = 0.5t_{gs}^* + 0.5t_{cb}^*$$

**Experiments**
Our team experimented extensively with models such as mDeBERTa, Aya101, Llama3, and Mistral Large; however, we ultimately chose Gemma2 27B (a decoder-only model version) because it achieved superior metrics results. We also tried Gemma3 27B, but for the downstream task finetuning - the training dynamics were the same as for the 2d version, so we didn't switch to it for following experiments. Our approach achieves SOTA results with a wide margin.

Almost all experiments were done in this exact or similar configuration:
```
lora_config:
  r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  target_modules: ["o_proj", "v_proj", "q_proj", "k_proj", "gate_proj", "down_proj", "up_proj"]

train_args:
  learning_rate: 0.00002
  weight_decay: 0.01
  lr_scheduler_type: "cosine"
  num_train_epochs: 5
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4
```

**Results**

Here you can see results of models from each fold (with val score in the submission name) and the final ensemble prediction:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1697396%2F696e67888b52bd2b7e18ecf2c4faf029%2FScreenshot%202025-04-03%20at%2021.17.05.png?generation=1743704288846098&alt=media)


# Span Identification

**Overview**

Our solution is out of fold ensemble prediction of the custom bidirectional encoder based on the Gemma2-27b (model for token classification) for 5 folds with custom threshold tuning specifically for F1 metric optimization. 

You can check the source code in the [repo](https://github.com/AntonBazdyrev/unlp2025_shared_task/tree/master)

**Validation**
Because we already knew the split for sure from the classification track - we used the same validation strategy in this task.

**Token classification problem statement**

We implemented a token classification labeling dataset using a custom binary-format labeling scheme without the BIO format, as our evaluation metric operates at the character level and not on the entity level, so we don't need to merge multi-token entities in order to maximize the metric.

**Thresholds Optimization**
You can check the more detailed analysis about threshold optimization for the F1 score [here](https://www.kaggle.com/competitions/unlp-2025-shared-task-classification-techniques/discussion/571513) in the Thresholds Optimization section. But in this task simple grid search optimization shows the best and consistent result.

**Experiments**
We explored various model architectures, including encoder-only models such as mBERT, XLM-RoBERTa, EuroBERT, and mDeBERTaV3—the latter proving to be the best-performing among “small” BERT-based architectures. However, we hypothesized that decoder-only LLMs, despite their uni-directional attention limitations, could achieve even better performance due to their larger parameter count and extensive pretraining. This hypothesis was confirmed through experimentation.

Ultimately, we developed a custom encoder-only model architecture that enables bidirectional attention for large LLMs. Given the strong performance of the decoder-only Gemma 2-27B in our previous experiments, we chose it as the foundation for our final model, although AYA-101 and mDeBERTa encoder-only performed nearly as well. We pre-trained a bidirectional version of Gemma 2-27B on MLM task on the [Ukrainian news dataset](https://huggingface.co/datasets/zeusfsx/ukrainian-news) and the [Russian news dataset](https://huggingface.co/datasets/AIR-Bench/qa_news_ru). After pre-training, we fine-tuned the model on the competition data.

Our custom bidirectional Gemma significantly outperformed both traditional encoder-only architectures and decoder-only LLMs, demonstrating the effectiveness of bidirectional attention in large-scale models.

**Results**
mDeBERTaV3:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1697396%2Ff1e7c488b08b64788d5ef04e31ea2f39%2Fmdeberta.png?generation=1743709099779638&alt=media)

AYA-101:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1697396%2Fa332610bb71dd1d040544a3fee9cc66e%2Faya101.png?generation=1743709118139630&alt=media)

Gemma 2-27B default decoder-only:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1697396%2Fe7a80b47a70e2c00beb766ef06d4bb60%2Fgemma27b.png?generation=1743709136217809&alt=media)

Gemma 2-27B mlm-pretrained bidirectional encoder:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1697396%2F51b4443b7fe11711748db8e7cf16cf50%2Fgemma27mlm.png?generation=1743709170697793&alt=media)

Ensemble of Gemma 2-27B mlm-pretrained bidirectional encoders:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1697396%2Ff265bfba270f0ed349fa39a83d6653d4%2Fensemble.png?generation=1743709201439096&alt=media)

## Legacy 
### Ideas to check

- EDA (Sasha)
- Metric understanding (Ivan Havlytskyi)
- Metric implementation for span ident (Ivan Bashtovyi)
- Translation for classification task (Artur)
- Auxiliary loss check on bert model (Ivan Havlytskyi?)
- Data scraping and pretrain of small llm / bert (unclear)
- Usage of 400b llm zero shot or finetune (Anton) - done

### Exp results
GPT4o
- zero shot: 0.32
- zero shot CoT: 0.36
