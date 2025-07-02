import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """ 
    This class implement the supervised contrastive loss.
    Consider for each image 
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def build_target_matrix(flags):
        """
        Costruisce la matrice target per SupConLoss.
    
        Args:
            flags: vettore con le etichette delle classi (es. [1,1,1,0,0,2,2])
    
        Returns:
            Matrice target dove target[i,j] = 1 se flags[i] == flags[j], 0 altrimenti
        """
        flags = torch.tensor(flags)
        # Espande i flags per confronto tra tutti gli elementi
        flags_expanded_row = flags.unsqueeze(1)
        flags_expanded_col = flags.unsqueeze(0)
    
        # Crea la matrice di similarità (1 se stessa classe, 0 altrimenti)
        target = (flags_expanded_row == flags_expanded_col).float()
    
        # Imposta a 0 gli elementi sulla diagonale (non confrontiamo un esempio con se stesso)
        #target.fill_diagonal_(0)
    
        return target

    def forward(self, image_embedding, text_embedding, flag):
        batch_size = image_embedding.shape[0]
        labels = self.build_target_matrix(flag)
        

        # caso multi-caption
        if len(text_embedding.shape) == 3:
            num_captions = text_embedding.shape[1]
            loss = 0

            for i in range(num_captions):
                text_emb = text_embedding[:, i, :]  # (B, D)
                #FOR IMAGE TO TEXT
                logits_i2t = torch.matmul(image_embedding, text_emb.T) / self.temperature
                exp_logits_i2t = torch.exp(logits_i2t)
                log_prob_i2t = logits_i2t - torch.log(exp_logits_i2t.sum(dim=1, keepdim=True))
                positive_mask_i2t = (labels > 0).float()
                per_instance_loss_i2t = - (log_prob_i2t * positive_mask_i2t).sum(dim=1) / torch.clamp(positive_mask_i2t.sum(dim=1), min=1.0)
                
                #FOR TEXT TO IMAGE
                logits_t2i = logits_i2t.T
                exp_logits_t2i = torch.exp(logits_t2i)
                log_prob_t2i = logits_t2i - torch.log(exp_logits_t2i.sum(dim=1, keepdim=True))
                positive_mask_t2i = (labels > 0).float()
                per_instance_loss_t2i = - (log_prob_t2i * positive_mask_t2i).sum(dim=1) / torch.clamp(positive_mask_t2i.sum(dim=1), min=1.0)

            loss = (per_instance_loss_i2t.mean() + per_instance_loss_t2i.mean()) / 2

        # caso single-caption
        else:
            #FOR IMAGE TO TEXT
            logits_i2t = torch.matmul(image_embedding, text_emb.T) / self.temperature
            exp_logits_i2t = torch.exp(logits_i2t)
            log_prob_i2t = logits_i2t - torch.log(exp_logits_i2t.sum(dim=1, keepdim=True))
            positive_mask_i2t = (labels > 0).float()
            per_instance_loss_i2t = - (log_prob_i2t * positive_mask_i2t).sum(dim=1) / torch.clamp(positive_mask_i2t.sum(dim=1), min=1.0)
                
            #FOR TEXT TO IMAGE
            logits_t2i = logits_i2t.T
            exp_logits_t2i = torch.exp(logits_t2i)
            log_prob_t2i = logits_t2i - torch.log(exp_logits_t2i.sum(dim=1, keepdim=True))
            positive_mask_t2i = (labels > 0).float()
            per_instance_loss_t2i = - (log_prob_t2i * positive_mask_t2i).sum(dim=1) / torch.clamp(positive_mask_t2i.sum(dim=1), min=1.0)

            loss = (per_instance_loss_i2t.mean() + per_instance_loss_t2i.mean()) / 2

        # accuracy t2i
        pred_t2i = torch.argmax(logits_t2i, dim=0)
        acc_t2i = (pred_t2i == labels).float().mean()

        return loss, acc_t2i
