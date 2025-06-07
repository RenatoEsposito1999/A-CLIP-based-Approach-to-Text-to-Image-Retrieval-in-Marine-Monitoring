import torch
import torch.nn as nn
import timm
import numpy as np
from transformers import AutoModel, CLIPModel
from lora import apply_lora_to_clip
from typing import Optional

class myClipModel(CLIPModel):
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.FloatTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = text_outputs.pooler_output

        return pooled_output

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.FloatTensor:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        pooled_output = vision_outputs.pooler_output

        return pooled_output
 
class RetrievalModel(nn.Module):
    def __init__(self, 
                 opts, 
                 embed_dim=512):
        super().__init__()
        '''# Vision Encoder
        self.vision_encoder = timm.create_model(opts.vision_encoder, pretrained=True)
        self.vision_encoder.reset_classifier(0)
        #self.freeze_module(self.vision_encoder)
        self.vision_encoder = apply_lora_to_vit(self.vision_encoder)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        vision_output_dim = self.vision_encoder.num_features
        # Text Encoder
        self.bert = AutoModel.from_pretrained(opts.text_encoder)
        #self.freeze_module(self.bert)
        self.bert = apply_lora_to_bert(self.bert)

        text_output_dim = self.bert.config.hidden_size'''
        
        #Definition of CLIP model
        self.clip_model = myClipModel.from_pretrained("openai/clip-vit-base-patch32")
        # Ora applica il freezing/unfreezing
        for param in self.clip_model.parameters():
            param.requires_grad = False
        if opts.lora:
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = True
                
            for param in self.clip_model.text_model.parameters():
                param.requires_grad = True
            self.clip_model = apply_lora_to_clip(self.clip_model)
            self.clip_model.print_trainable_parameters()
        vision_output_dim = self.clip_model.vision_embed_dim
        text_output_dim = self.clip_model.text_embed_dim
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # Projection heads
        self.image_proj = self.build_mlp(vision_output_dim, embed_dim)
        self.text_proj = self.build_mlp(text_output_dim, embed_dim)
        
    

    def freeze_module(self,module):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def build_mlp(self, input_dim, output_dim, hidden_dim=1024, dropout=0.1):
        '''return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )'''
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
 
    def encode_image(self, x):
        with torch.no_grad():
            features = self.clip_model.get_image_features(x)
        return self.image_proj(features)
 
    def encode_text(self, text_inputs, attention_mask):
        with torch.no_grad():
            features = self.clip_model.get_text_features(input_ids=text_inputs, attention_mask=attention_mask)
        return self.text_proj(features)
 
    def forward(self, images, text_inputs, attention_mask):
        image_embeds = self.encode_image(images)  # (B, D)
        text_embeds = self.encode_text(text_inputs=text_inputs, attention_mask=attention_mask)  # (B, D)

        # normalize
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
 
        return image_embeds, text_embeds, self.logit_scale
    


