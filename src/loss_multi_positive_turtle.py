import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """ 
    symmetric cross-entropy contrastive loss. 
    It aims to maximize the cosine similarity 
    between matched image-text pairs while 
    minimizing it for unmatched pairs within a batch
    
    This version supports multiple captions per image.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embedding, text_embedding, flag):
        batch_size = image_embedding.shape[0]
        labels = torch.arange(batch_size, device=image_embedding.device)

        # caso multi-caption
        if len(text_embedding.shape) == 3:
            num_captions = text_embedding.shape[1]
            total_loss = 0.0

            for i in range(num_captions):
                text_emb = text_embedding[:, i, :]  # (B, D)
                logits = torch.matmul(image_embedding, text_emb.T) / self.temperature
               
                # masking: evita che sample con stesso flag (non 0) siano negativi tra loro
                flag_i = flag.view(-1, 1)  # (B, 1)
                same_class = (flag_i == flag_i.T)  # (B, B)
                mask = same_class & (flag_i != 0)
                # evita masking sul target corretto
                mask[torch.arange(batch_size), labels] = False
                
                logits = logits.masked_fill(mask, float('-inf'))
               
                rows_all_inf = torch.isinf(logits).all(dim=1)
                if rows_all_inf.any():
                    print("Attenzione: alcune righe di logits sono tutte -inf")
                # contrastive losses
                loss_i2t = F.cross_entropy(logits, labels)
                loss_t2i = F.cross_entropy(logits.T, labels)
                total_loss += (loss_i2t + loss_t2i) / 2

            total_loss /= num_captions

        # caso single-caption
        else:
            logits = torch.matmul(image_embedding, text_embedding.T) / self.temperature
            flag_i = flag.view(-1, 1)
            same_class = (flag_i == flag_i.T)
            mask[torch.arange(batch_size), labels] = False     
            logits = logits.masked_fill(mask, float('-inf'))   
            rows_all_inf = torch.isinf(logits).all(dim=1)
            if rows_all_inf.any():
                print("Attenzione: alcune righe di logits sono tutte -inf")
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            total_loss = (loss_i2t + loss_t2i) / 2

        # accuracy t2i
        pred_t2i = torch.argmax(logits, dim=0)
        acc_t2i = (pred_t2i == labels).float().mean()

        return total_loss, acc_t2i
