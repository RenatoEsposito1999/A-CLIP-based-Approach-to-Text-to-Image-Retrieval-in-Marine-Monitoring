from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel


def apply_lora_to_clip(clip_model, r=8, alpha=16, dropout=0.1):
    config = LoraConfig(
        #task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "visual_projection", "text_projection"]
    )
    return get_peft_model(clip_model, config)