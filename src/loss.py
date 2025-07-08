import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

def build_target_matrix(flags):
        """
        Costruisce la matrice target per SupConLoss.
        Args:
            flags: vettore con le etichette delle classi (es. [1,1,1,0,0,2,2])
    
        Returns:
            Matrice target dove target[i,j] = 1 se flags[i] == flags[j], 0 altrimenti
        """
        #print(flags)
        '''number_categories = defaultdict(int)
        for x in flags.tolist():
            if x < 0:
                number_categories[x] += 1
        print(number_categories)
        print("BATCH SIZE: ", len(flags.tolist()))'''
        
        # Espande i flags per confronto tra tutti gli elementi
        flags_expanded_row = flags.unsqueeze(1)
        flags_expanded_col = flags.unsqueeze(0)
       
        
        # Crea la matrice di similarità (1 se stessa classe, 0 altrimenti)
        target = (flags_expanded_row == flags_expanded_col).float()
        
      
        # Imposta a 0 gli elementi sulla diagonale (non confrontiamo un esempio con se stesso)
        #target.fill_diagonal_(0)
        return target

def contrastiveLoss(image_embedding, text_embedding, cats, temperature=None):
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
    batch_size = image_embedding.shape[0]
    if temperature:
        temperature = torch.clamp(temperature.exp(), max=100)
        temperature = 1/temperature
    else:
        temperature = 0.07
    labels = build_target_matrix(cats)   
    #DECOMMENT THE LINES ABOVE FOR CHECKING IF THE TEXT AND IMAGE ARE NORMALIZED 
    '''# Calcola le norme L2 per ogni embedding nel batch (lungo l'asse D)
    text_norms = torch.norm(text_embedding, p=2, dim=1)  # shape: (B,)
    image_norms = torch.norm(image_embedding, p=2, dim=1)
    # Verifica se tutti sono normalizzati (con tolleranza numerica)
    print("Text embeddings normalizzati?", torch.allclose(text_norms, torch.ones_like(text_norms), rtol=1e-3))
    print("Image embeddings normalizzati?", torch.allclose(image_norms, torch.ones_like(image_norms), rtol=1e-3))
    exit()'''
    positive_mask = (labels > 0).float()
    #FOR IMAGE TO TEXT
    logits_i2t = torch.matmul(image_embedding, text_embedding.T) / temperature
    log_prob_i2t = F.log_softmax(logits_i2t, dim=1)
    per_instance_loss_i2t = - (log_prob_i2t * positive_mask).sum(dim=1) / torch.clamp(positive_mask.sum(dim=1), min=1.0)
    
    #FOR TEXT TO IMAGE
    logits_t2i = torch.matmul(text_embedding,image_embedding.T) / temperature
    log_prob_t2i = F.log_softmax(logits_t2i, dim=1)
    per_instance_loss_t2i = - (log_prob_t2i * positive_mask).sum(dim=1) / torch.clamp(positive_mask.sum(dim=1), min=1.0)
    
    loss = (per_instance_loss_i2t.mean() + per_instance_loss_t2i.mean()) / 2

    # accuracy t2i
    #pred_t2i = torch.argmax(logits_t2i, dim=0)
    #acc_t2i = (pred_t2i == labels).float().mean()

    return loss

