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
    
    
def contrastive_loss(image_embeds, text_embeds, logit_scale, turtle):
    logit_scale = logit_scale.exp()
    logits = logit_scale * image_embeds @ text_embeds.T  # (B, B)
    # Step 1: Crea classi
    labels = []
    class_id = 1
    #turtle = [1,0,0,1,0]
    for is_turtle in turtle:
        if is_turtle.item() == 1:
            labels.append(0)
        else:
            labels.append(class_id)
            class_id += 1
    labels = torch.tensor(labels, device=logits.device)
    #labels = [0,1,2,0,3]
    
    

    # Step 2: positive_mask
    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    
    '''# Step 3: Calcolo loss
    sim_i2t = torch.nn.functional.log_softmax(logits, dim=1)
    loss_i2t = - (positive_mask * sim_i2t).sum() / positive_mask.sum()'''

    sim_t2i = torch.nn.functional.log_softmax(logits.T, dim=1)
    print(sim_t2i)
    loss_t2i = - (positive_mask.T * sim_t2i).sum() / positive_mask.T.sum()
    return loss_t2i

    #return (loss_i2t + loss_t2i) / 2
    #return (loss_i2t + loss_t2i) / 2