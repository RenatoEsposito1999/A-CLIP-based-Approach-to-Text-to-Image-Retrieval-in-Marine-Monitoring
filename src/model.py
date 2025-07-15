import torch
import torch.nn as nn
from transformers import CLIPModel
from peft import get_peft_model, LoraConfig


class CLIP_model(nn.Module):
    def __init__(self, model_name:str):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16)
        '''for param in clip_model.vision_model.parameters():
                param.requires_grad = True
        for param in clip_model.text_model.parameters():
            param.requires_grad = True'''
        self.clip_model = self.apply_lora_to_clip()
        self.clip_model.logit_scale.requires_grad = True
        self.clip_model.print_trainable_parameters()
        

    def apply_lora_to_clip(self, r=8, alpha=16, dropout=0.1):
        config = LoraConfig(
            #task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "visual_projection", "text_projection"]
            #target_modules=["visual_projection", "text_projection"]
        )
        return get_peft_model(self.clip_model, config)
    
    def forward(self, images, text_inputs, attention_masks):
        output = self.clip_model(input_ids=text_inputs, pixel_values=images, attention_mask=attention_masks, return_loss=False)
        return output.image_embeds, output.text_embeds, self.clip_model.logit_scale


    

