import torch
import torch.nn as nn

def masked_contrastive_loss(image_embeds, text_embeds, labels, temperature=0.07):
    """
    image_embeds: (B, D)
    text_embeds: (B, D)
    labels: (B,) categoria (es. 0 = turtle, 1 = dolphin, ecc.)
    """
    unique_id_coco = 1
    category_id = []
    # TO CHECK RESPECT TO CATEGORY INFO.JSON in dataset folder
    for id in labels:
        if id == 1:
            category_id.append(-2) #turtle
        elif id == 7:
            category_id.append(-5) #sea
        elif id == 13:
            category_id.append(-10) #dolphin
        elif id == 14:
            category_id.append(-11) #debris
        else:
            category_id.append(unique_id_coco)
            unique_id_coco += 1
    labels = torch.tensor(category_id)
    device = image_embeds.device
    B = image_embeds.size(0)
    logit_scale = torch.clamp(temperature.exp(), max=100)
    # Normalize
    image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

    # Compute similarity logits
    logits = logit_scale * image_embeds @ text_embeds.T  # (B, B)

    # Ground truth labels (i.e., position i is the correct match)
    target = torch.arange(B, device=device)

    # Costruisci maschera di falsi negativi (stessa categoria ma posizione ≠ i)
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, B]
    not_self = ~torch.eye(B, dtype=torch.bool, device=device)
    false_neg_mask = label_matrix.to(device) & not_self.to(device=device)  # stesse categorie ≠ se stessi
    '''print("\n ", labels)
    print("\n ", label_matrix)
    print("\n ", not_self)
    print("\n ", false_neg_mask)'''
    
    # Maschera i falsi negativi nei logits (image-to-text e text-to-image)
    logits_i2t = logits.masked_fill(false_neg_mask, float('-inf'))
    logits_t2i = logits.T.masked_fill(false_neg_mask.T, float('-inf'))
    #print("\n ", logits_i2t)
    # CLIP-style simmetrica
    loss_i2t = nn.functional.cross_entropy(logits_i2t, target)
    loss_t2i = nn.functional.cross_entropy(logits_t2i, target)
    return (loss_i2t + loss_t2i) / 2

def contrastive_loss(image_embeds, text_embeds, logit_scale=0.07):
    logit_scale = logit_scale.exp()
    logits = logit_scale * image_embeds @ text_embeds.T  # (B, B)
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.T, labels)
    #return loss_t2i
    return (loss_i2t + loss_t2i) / 2


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
    # TO CHECK RESPECT TO CATEGORY INFO.JSON in dataset folder
    for id in labels:
        if id == 1:
            category_id.append(-2) #turtle
        elif id == 7:
            category_id.append(-5) #sea
        elif id == 13:
            category_id.append(-10) #dolphin
        elif id == 14:
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
