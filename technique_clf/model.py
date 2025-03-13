import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

def get_model(config, num_labels, load_lora=True):
    pretrained_model = config['model']['pretrained_model']
    config['nf4_config']["bnb_4bit_compute_dtype"] = getattr(
        torch, config['nf4_config']["bnb_4bit_compute_dtype"].split(".")[-1]
    )
    bnb_config = BitsAndBytesConfig(
        **config['nf4_config']
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )

    model = prepare_model_for_kbit_training(model)

    if load_lora:
        lora_config = LoraConfig(
            **config['lora_config']
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model
