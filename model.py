import torch
import torch.nn as nn
import timm
from transformers import BertModel, AutoModel
import open_clip
import numpy as np
 
class RetrievalModel(nn.Module):
    def __init__(self, 
                 opts, 
                 embed_dim=512):
        super().__init__()
        # Vision Encoder
        self.vision_encoder = timm.create_model(opts.vision_encoder, pretrained=True)
        self.vision_encoder.reset_classifier(0)
        self.freeze_module(self.vision_encoder)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        vision_output_dim = self.vision_encoder.num_features
        # Text Encoder
        self.bert = AutoModel.from_pretrained(opts.text_encoder)
        self.freeze_module(self.bert)
        text_output_dim = self.bert.config.hidden_size
        # Projection heads
        self.image_proj = self.build_mlp(vision_output_dim, embed_dim)
        self.text_proj = self.build_mlp(text_output_dim, embed_dim)
    

    def freeze_module(self,module):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def build_mlp(self, input_dim, output_dim, hidden_dim=1024, dropout=0.1):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
 
    def encode_image(self, x):
        with torch.no_grad():
            features = self.vision_encoder(x)
        return self.image_proj(features)
 
    def encode_text(self, text_inputs):
        with torch.no_grad():
            outputs = self.bert(**text_inputs)
            x = outputs.pooler_output
        return self.text_proj(x)
 
    def forward(self, images, text_inputs):
        image_embeds = self.encode_image(images)  # (B, D)
        text_embeds = self.encode_text(text_inputs)  # (B, D)
        logit_scale = torch.clamp(self.logit_scale, max=100)  # Max=100 è un valore sicuro
        # normalize
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
 
        return image_embeds, text_embeds, logit_scale
    


