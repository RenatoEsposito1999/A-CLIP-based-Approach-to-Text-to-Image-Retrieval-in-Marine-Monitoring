import torch
from transformers import get_cosine_schedule_with_warmup

def get_optimizer_and_scheduler(model, lr,weight_decay, tot_num_epochs, steps_per_epoch):
    """
    Define the optimizer and the learning rate scheduler.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_params = [
        {"params": trainable_params, "lr": lr, "weight_decay": weight_decay},
    ]
    optimizer = torch.optim.AdamW(optimizer_params)
        
    #START DEFINITION COSINE SCHEDULER
    warmup_ratio = 0.1
    total_training_steps = tot_num_epochs * steps_per_epoch
    num_warmup_steps = int(warmup_ratio*total_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )
        
    return optimizer, scheduler