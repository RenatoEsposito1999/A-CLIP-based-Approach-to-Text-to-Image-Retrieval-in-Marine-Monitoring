import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
focus_id = []
with open("/workspace/text-to-image-retrivial/datasets/annotations/category_info.json", "r") as f:
            data = json.load(f)
            focus_id.append(data["turtle"][0])
            focus_id.append(data["dolphin"][0])
            focus_id.append(data["sea"][0])
            focus_id.append(data["debris"][0])

def train(model,dataloader,n_epochs, loss_fn,device, optimizer, scheduler, writer, val_dataloader):
    model = model.to(device)
    for epoch in tqdm(range(n_epochs)):
        model.train()
        total_loss = 0
        for batch in dataloader:
            images, captions, masks, cats = batch
            images = images.to(device)
            captions = captions.to(device)
            masks = masks.to(device)
            cats = cats.to(device)
            img_embs,text_embs, logit_scale=model(images,captions,masks)
            
            loss = loss_fn(img_embs, text_embs, cats, logit_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
    
            scheduler.step()
        writer.add_scalar("Loss/train", total_loss / len(dataloader), epoch+1)

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(dataloader):.4f}")
        # 2. Norma dei gradienti (se sono ~0, il modello non impara)
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item()
        print(f"Gradient Norm: {total_grad_norm}")
        #validation(dataloader=val_dataloader, model=model,writer=writer, train_epoch=epoch, device=device)

def validation(dataloader, model,writer, train_epoch, device):
    model.eval()
    all_img_embs, all_text_embs, all_cats = [], [],[]
    for batch in dataloader:
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                images, captions, masks, cats = batch
                images = images.to(device)
                captions = captions.to(device)
                masks = masks.to(device)
                cats = cats.to(device)
                img_embs,text_embs=model(images,captions,masks)
                all_img_embs.append(img_embs)
                all_text_embs.append(text_embs)
                all_cats.append(cats)
                #loss, _ = loss_fn(text_embs,img_embs,cats)
    all_img_embs = torch.cat(all_img_embs, dim=0)
    all_text_embs = torch.cat(all_text_embs, dim=0)
    all_cats = torch.cat(all_cats,dim=0)
    compute_metrics(writer=writer,image_embeddings=all_img_embs,epoch=train_epoch, text_embeddings=all_text_embs, categories=all_cats)

def compute_metrics(writer, text_embeddings, image_embeddings, epoch,k_values=[1, 5, 10], categories=None):
    text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_embeddings_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    
    sim_matrix = torch.mm(text_embeddings_norm, image_embeddings_norm.t())  #[N_text, N_images]
    
    # Ottieni gli indici top-k (discendenti)
    top_k_indices = torch.topk(sim_matrix, k=max(k_values), dim=1).indices  # [N_text, max(k_values)]

    results = {}
    n_total = len(text_embeddings)
    
    # 1. Calcolo GLOBALE (tutto il dataset)
    for k in k_values:
        # Ottieni le categorie retrieve per ogni query
        retrieved_cats = categories[top_k_indices[:, :k]]  # [N_text, k]
        query_cats = categories.unsqueeze(1)  # [N_text, 1]
        
        # Controlla se la categoria query è tra quelle retrieve
        correct = (retrieved_cats == query_cats).any(dim=1).sum().item()
        results[f"cat_all_R@{k}"] = correct / n_total
        writer.add_scalar(f"cat_all_R@{k}", results[f"cat_all_R@{k}"], epoch+1)

    
    # 2. Calcolo FOCUS (solo categorie specificate)
    focus_mask = torch.isin(categories, torch.tensor(focus_id, device=categories.device))
    focus_indices = torch.where(focus_mask)[0]
    n_focus = len(focus_indices)
        
    for k in k_values:
        if n_focus > 0:
            retrieved_cats_focus = categories[top_k_indices[focus_indices, :k]]  # [N_focus, k]
            query_cats_focus = categories[focus_indices].unsqueeze(1)  # [N_focus, 1]
            correct = (retrieved_cats_focus == query_cats_focus).any(dim=1).sum().item()
            results[f"cat_focus_R@{k}"] = correct / n_focus
        else:
            results[f"cat_focus_R@{k}"] = 0.0
        writer.add_scalar(f"cat_focus_R@{k}", results[f"cat_focus_R@{k}"], epoch+1)
    return results