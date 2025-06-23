import torch
import torch.nn as nn
'''def contrastive_loss(image_embeds, text_embeds, logit_scale=0.07):
    logit_scale = logit_scale.exp()
    logits = logit_scale * image_embeds @ text_embeds.T  # (B, B)
    labels = torch.arange(len(logits)).to(logits.device)
    #loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.T, labels)
    return loss_t2i
    #return (loss_i2t + loss_t2i) / 2'''

'''def supcon_loss(anchor, positives, labels, temperature):
    """
    anchor: Tensor [N, D] <- Text embeds
    positives: Tensor [N, D] <- Img embeds
    labels: Tensor [N] <-- Category id
    """
    device = anchor.device
    labels = labels.to(device)
    #temperature = temperature.exp()
    temperature = torch.clamp(temperature.exp(), max=100)  # <-- Gradient-friendly

    # Calcola similarità
    #sim = torch.matmul(anchor,positives.T) / temperature

    anchor = anchor / anchor.norm(dim=1, keepdim=True)
    positives = positives / positives.norm(dim=1, keepdim=True)

    sim = temperature * torch.matmul(anchor, positives.T)  # [N, N]

    # Mask dei positivi (stessa label ma non self)
    
    mask = labels.unsqueeze(1) == labels.unsqueeze(0)      # [N, N]
    
    log_probs = torch.nn.functional.log_softmax(sim, dim=1)
    # Solo i positivi contribuiscono
    mean_log_prob_pos = (mask.float() * log_probs).sum(1) / mask.sum(1).clamp(min=1)

    # Final loss
    loss = -mean_log_prob_pos.mean()

    sim = torch.matmul(anchor, positives.T)
   
    
    
    return loss'''



def supcon_loss(anchor, positives, labels, temperature):
    """
    Modified SupCon loss that excludes same-category pairs (except diagonal) from negatives.
    
    Args:
        anchor: Tensor [N, D] <- Text embeddings (normalized)
        positives: Tensor [N, D] <- Image embeddings (normalized)
        labels: Tensor [N] <- Category IDs
        temperature: Logit scale (will be exp() clamped)
    """
    unique_id_coco = 1
    category_id = []
    for id in labels:
        if id == 2:
            category_id.append(-2) #turtle
        elif id == 5:
            category_id.append(-5) #sea
        elif id == 10:
            category_id.append(-10) #dolphin
        elif id == 11:
            category_id.append(-11) #debris
        else:
            category_id.append(unique_id_coco)
            unique_id_coco += 1
    labels = torch.tensor(category_id)
    device = anchor.device
    N = anchor.size(0)
    
    # Normalize embeddings and compute similarity
    anchor = anchor / anchor.norm(dim=1, keepdim=True)
    positives = positives / positives.norm(dim=1, keepdim=True)
    
    # Apply temperature (gradient-safe)
    temperature = torch.clamp(temperature.exp(), max=100)
    sim = temperature * torch.matmul(anchor, positives.T)  # [N, N]
    
    # Create mask for same-category pairs (excluding diagonal)
    same_category = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(device)  # [N, N]
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)   # Mask for non-diagonal
    mask_to_exclude = same_category & diag_mask                  # Same cat. but not self
    # Set excluded similarities to -inf before log_softmax
    masked_sim = sim.masked_fill(mask_to_exclude, float('-inf'))
   
    # Positive pairs are the diagonal (anchor <-> positive)
    pos_mask = torch.eye(N, dtype=torch.bool, device=device)
    
    # Compute log-probs and focus only on positive pairs (diagonal)
    log_probs = torch.nn.functional.log_softmax(masked_sim, dim=1)
    mean_log_prob_pos = log_probs[pos_mask].mean()
    # Final loss (negative log-likelihood of positive pairs)
    loss = -mean_log_prob_pos
    
    return loss
