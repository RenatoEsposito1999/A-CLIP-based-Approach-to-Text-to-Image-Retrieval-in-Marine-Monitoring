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

def supcon_loss(anchor, positives, labels, temperature):
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
    pos_scores = sim[torch.arange(len(labels)), torch.arange(len(labels))]
    print("Mean positive similarity:", pos_scores.mean().item())
    
    return loss

