import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoModelForTokenClassification
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

import torch

from packaging import version
import importlib.metadata

from transformers import Gemma2Model, Gemma2ForCausalLM, Gemma2PreTrainedModel, Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2Attention,
    Gemma2FlashAttention2,
    Gemma2SdpaAttention,
    Gemma2MLP,
    Gemma2RMSNorm,
)

from torch import nn
from transformers.utils import logging

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.import_utils import _is_package_available
from transformers.cache_utils import Cache, StaticCache
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.gemma2.modeling_gemma2 import *

from peft import PeftModel

logger = logging.get_logger(__name__)

def is_transformers_attn_greater_or_equal_4_41():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.41.0"
    )


class ModifiedGemma2Attention(Gemma2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedGemma2FlashAttention2(Gemma2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedGemma2SdpaAttention(Gemma2SdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


GEMMA2_ATTENTION_CLASSES = {
    "eager": ModifiedGemma2Attention,
    "flash_attention_2": ModifiedGemma2FlashAttention2,
    "sdpa": ModifiedGemma2SdpaAttention,
}


class ModifiedGemma2DecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = GEMMA2_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_sliding = not bool(layer_idx % 2)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window



class Gemma2BiModel(Gemma2Model):
    _no_split_modules = ["ModifiedGemma2DecoderLayer"]

    def __init__(self, config: Gemma2Config):
        Gemma2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedGemma2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if past_key_values is not None:
            target_length = past_key_values.get_max_length()
        else:
            target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.zeros(
                (sequence_length, target_length), dtype=dtype, device=device
            )
            # causal_mask = torch.full(
            #     (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            # )
            # if sequence_length != 1:
            #     causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class Gemma2BiForMNTP(Gemma2ForCausalLM):
    def __init__(self, config):
        Gemma2PreTrainedModel.__init__(self, config)
        self.model = Gemma2BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)


class Gemma2BiMLM(Gemma2ForCausalLM):
    def __init__(self, config):
        Gemma2PreTrainedModel.__init__(self, config)
        self.model = Gemma2BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)


def get_model(config, load_lora=True):
    pretrained_model = config['model']['pretrained_model']
    mock_model = config['model']['mock_clf_model']
    config['nf4_config']["bnb_4bit_compute_dtype"] = getattr(
        torch, config['nf4_config']["bnb_4bit_compute_dtype"].split(".")[-1]
    )
    bnb_config = BitsAndBytesConfig(
        **config['nf4_config']
    )

    id2label = {0: 0, 1: 1}
    label2id = {0: 0, 1: 1}
    
    if config['model']['architecture'] == 'mdeberta':
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model,
            id2label=id2label,
            label2id=label2id
        )
    else: 
        backbone_model = Gemma2BiModel.from_pretrained(
            pretrained_model, torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
    
        id2label = {0: 0, 1: 1}
        label2id = {0: 0, 1: 1}
        model = AutoModelForTokenClassification.from_pretrained(
            mock_model,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            id2label=id2label,
            label2id=label2id
        )
        model.model = backbone_model
        torch.cuda.empty_cache()
    
        model = prepare_model_for_kbit_training(model)
    
        if load_lora:
            lora_config = LoraConfig(
                **config['lora_config']
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    
    return model
