import torch
import torch.nn as nn
import torch.nn.functional as F

def build_target_matrix(flags):
        """
        Costruisce la matrice target per SupConLoss.
        Args:
            flags: vettore con le etichette delle classi (es. [1,1,1,0,0,2,2])
    
        Returns:
            Matrice target dove target[i,j] = 1 se flags[i] == flags[j], 0 altrimenti
        """
        #flags = torch.tensor(flags)
        '''print(flags)
        print([[x,flags.tolist().count(x)] for x in set(flags.tolist())])'''
    
        # Espande i flags per confronto tra tutti gli elementi
        flags_expanded_row = flags.unsqueeze(1)
        flags_expanded_col = flags.unsqueeze(0)
    
        # Crea la matrice di similarità (1 se stessa classe, 0 altrimenti)
        target = (flags_expanded_row == flags_expanded_col).float()
        '''print(target)
        exit()'''
        # Imposta a 0 gli elementi sulla diagonale (non confrontiamo un esempio con se stesso)
        #target.fill_diagonal_(0)
        return target

def contrastiveLoss(image_embedding, text_embedding, cats, temperature=0.07):
    batch_size = image_embedding.shape[0]
    labels = build_target_matrix(cats)    
    # caso single-caption
    #FOR IMAGE TO TEXT
    # Calcola le norme L2 per ogni embedding nel batch (lungo l'asse D)
    text_norms = torch.norm(text_embedding, p=2, dim=1)  # shape: (B,)
    image_norms = torch.norm(image_embedding, p=2, dim=1)
    # Verifica se tutti sono normalizzati (con tolleranza numerica)
    '''print("Text embeddings normalizzati?", torch.allclose(text_norms, torch.ones_like(text_norms), rtol=1e-3))
    print("Image embeddings normalizzati?", torch.allclose(image_norms, torch.ones_like(image_norms), rtol=1e-3))
    exit()'''
    logits_i2t = torch.matmul(image_embedding, text_embedding.T) / temperature
    exp_logits_i2t = torch.exp(logits_i2t)
    #log_prob_i2t = logits_i2t - torch.log(exp_logits_i2t.sum(dim=1, keepdim=True))
    log_prob_i2t = F.log_softmax(logits_i2t, dim=1)
    positive_mask_i2t = (labels > 0).float()
    per_instance_loss_i2t = - (log_prob_i2t * positive_mask_i2t).sum(dim=1) / torch.clamp(positive_mask_i2t.sum(dim=1), min=1.0)
    
    #FOR TEXT TO IMAGE
    logits_t2i = torch.matmul(text_embedding,image_embedding.T) / temperature
    #exp_logits_t2i = torch.exp(logits_t2i)
    #log_prob_t2i = logits_t2i - torch.log(exp_logits_t2i.sum(dim=1, keepdim=True))
    log_prob_t2i = F.log_softmax(logits_t2i, dim=1)
    positive_mask_t2i = (labels > 0).float()
    per_instance_loss_t2i = - (log_prob_t2i * positive_mask_t2i).sum(dim=1) / torch.clamp(positive_mask_t2i.sum(dim=1), min=1.0)
    loss = (per_instance_loss_i2t.mean() + per_instance_loss_t2i.mean()) / 2
    #loss = per_instance_loss_t2i.mean()

    # accuracy t2i
    #pred_t2i = torch.argmax(logits_t2i, dim=0)
    #acc_t2i = (pred_t2i == labels).float().mean()

    return loss

