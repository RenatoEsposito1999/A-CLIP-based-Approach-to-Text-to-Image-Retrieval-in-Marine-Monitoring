from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel
import timm

def apply_lora_to_bert(bert_model, r=8, alpha=16, dropout=0.1):
    config = LoraConfig(
        #task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value"]  # per BERT standard
    )
    return get_peft_model(bert_model, config)

def apply_lora_to_vit(vit_model, r=8, alpha=16, dropout=0.1):
    config = LoraConfig(
        #task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv"]  # tipico per ViT
    )
    return get_peft_model(vit_model, config)