import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

def build_target_matrix(flags):
        """
        Builds the target matrix for SupConLoss.

        Args:
            flags: A vector containing class labels (e.g., [1,1,1,0,0,2,2]).

        Returns:
            Target matrix where target[i,j] = 1 if flags[i] == flags[j], 0 otherwise.
        """
        
        
        #print(flags)
        '''number_categories = defaultdict(int)
        for x in flags.tolist():
            if x < 0:
                number_categories[x] += 1
        print(number_categories)'''
        #print("BATCH SIZE: ", len(flags.tolist()))
        
       
        flags_expanded_row = flags.unsqueeze(1)
        flags_expanded_col = flags.unsqueeze(0)
       
        
        # Create the matrix target where the corresponding entry with the same category is putted 1 otherwise 0
        target = (flags_expanded_row == flags_expanded_col).float()
        return target

def UniLoss(image_embedding, text_embedding, cats, temperature=None):
    """
    Compute the UniLoss (bidirectional contrastive loss) as used by Microsoft.
    
    Args:
        image_embedding: Tensor of shape [batch_size, embedding_dim]
        text_embedding: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size, batch_size] indicating positive pairs
        temperature: Softmax temperature parameter
        
    Returns:
        Scalar loss value
    """
    if temperature:
        temperature = torch.clamp(temperature.exp(), max=100)
        temperature = 1/temperature
    else:
        temperature = 0.07
    labels = build_target_matrix(cats)   
    #DECOMMENT THE LINES ABOVE FOR CHECKING IF THE TEXT AND IMAGE ARE NORMALIZED 
    '''# Calculate L2 norms for each embedding in the batch (along dimension D)
    text_norms = torch.norm(text_embedding, p=2, dim=1)  # shape: (B,)
    image_norms = torch.norm(image_embedding, p=2, dim=1)
    # Check if all embeddings are normalized (with numerical tolerance) 
    print("Text embeddings normalizzati?", torch.allclose(text_norms, torch.ones_like(text_norms), rtol=1e-3))
    print("Image embeddings normalizzati?", torch.allclose(image_norms, torch.ones_like(image_norms), rtol=1e-3))
    '''
    positive_mask = (labels > 0).float()
    #FOR IMAGE TO TEXT
    logits_i2t = torch.matmul(image_embedding, text_embedding.T) / temperature
    log_prob_i2t = F.log_softmax(logits_i2t, dim=1)
    #In the paper also called soft-crossentropy
    per_instance_loss_i2t = - (log_prob_i2t * positive_mask).sum(dim=1) / torch.clamp(positive_mask.sum(dim=1), min=1.0)
    
    #FOR TEXT TO IMAGE
    logits_t2i = torch.matmul(text_embedding,image_embedding.T) / temperature
    log_prob_t2i = F.log_softmax(logits_t2i, dim=1)
    #In the paper also called soft-crossentropy
    per_instance_loss_t2i = - (log_prob_t2i * positive_mask).sum(dim=1) / torch.clamp(positive_mask.sum(dim=1), min=1.0)
    
    loss = (per_instance_loss_i2t.mean() + per_instance_loss_t2i.mean()) / 2

    # accuracy t2i
    #pred_t2i = torch.argmax(logits_t2i, dim=0)
    #acc_t2i = (pred_t2i == labels).float().mean()

    return loss


def contrastiveLoss(image_embedding, text_embedding, temperature=None):
    batch_size = image_embedding.shape[0]
    if temperature:
        temperature = torch.clamp(temperature.exp(), max=100)
        temperature = 1/temperature
    else:
        temperature = 0.07
        
    # we create dummy labels for the batch.
    labels = torch.arange(batch_size, device=image_embedding.device)
    
    logits = torch.matmul(image_embedding, text_embedding.T) / temperature
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    total_loss = (loss_i2t + loss_t2i) / 2   

    return total_loss


def compute_loss(image_embedding, text_embedding, cats, temperature=None):
    uniloss = UniLoss(image_embedding=image_embedding, text_embedding=text_embedding, cats=cats, temperature=temperature)
    contrastive_loss = contrastiveLoss(image_embedding=image_embedding, text_embedding=text_embedding, temperature=temperature)
    
    total_loss = (uniloss + contrastive_loss) / 2
    
    return total_loss, uniloss, contrastive_loss
