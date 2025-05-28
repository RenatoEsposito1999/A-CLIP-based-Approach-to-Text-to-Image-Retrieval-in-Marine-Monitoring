import torch
import torch.nn as nn
import timm
from transformers import BertModel, BertTokenizer
import open_clip
 
class RetrievalModel(nn.Module):
    def __init__(self, 
                 vision_encoder="vit_base_patch16_224",
                 text_encoder="clip", 
                 embed_dim=512):
        super().__init__()
        # Vision Encoder
        self.vision_encoder = timm.create_model(vision_encoder, pretrained=True)
        self.vision_encoder.reset_classifier(0)
        self.vision_encoder.eval()
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
 
        vision_output_dim = self.vision_encoder.num_features
        # Text Encoder
        if text_encoder == "clip":
            model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            self.text_encoder = model.token_embedding
            self.text_transformer = model.transformer
            self.positional_embedding = model.positional_embedding
            self.ln_final = model.ln_final
            self.text_projection = model.text_projection
            self.text_encoder_type = "clip"
        elif text_encoder == "bert":
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            for p in self.bert.parameters():
                p.requires_grad = False
            self.text_encoder_type = "bert"
            text_output_dim = self.bert.config.hidden_size
        else:
            raise ValueError("Unsupported text encoder")
        # Projection heads
        self.image_proj = nn.Linear(vision_output_dim, embed_dim)
        if text_encoder == "clip":
            self.text_proj = nn.Identity()  # proiezione già fatta da CLIP
        else:
            self.text_proj = nn.Linear(text_output_dim, embed_dim)
 
    def encode_image(self, x):
        with torch.no_grad():
            features = self.vision_encoder(x)
        return self.image_proj(features)
 
    def encode_text(self, text_inputs):
        if self.text_encoder_type == "clip":
            with torch.no_grad():
                x = self.text_encoder(text_inputs) + self.positional_embedding
                x = self.text_transformer(x)
                x = x[torch.arange(x.shape[0]), text_inputs.argmax(dim=-1)]
                x = self.ln_final(x)
                x = x @ self.text_projection
        else:
            with torch.no_grad():
                outputs = self.bert(**text_inputs)
                x = outputs.pooler_output
        return self.text_proj(x)
 
    def forward(self, images, text_inputs):
        image_embeds = self.encode_image(images)  # (B, D)
        text_embeds = self.encode_text(text_inputs)  # (B, D)
 
        # normalize
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
 
        return image_embeds, text_embeds