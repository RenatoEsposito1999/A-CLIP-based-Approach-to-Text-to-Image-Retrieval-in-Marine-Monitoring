import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
import torch.nn.functional as F

import torch.nn.functional as F
best_val_loss = float("inf")
best_recall_5_focuss = -float("inf")
best_recall_5_all = -float("inf")
focus_ids = [-2]

def train(model,dataloader,n_epochs, loss_fn,device, optimizer, scheduler, writer, val_dataloader):
    model = model.to(device)
    for epoch in tqdm(range(n_epochs)):
        model.train()
        total_loss = 0
        total_uni = 0 
        total_contrastive = 0 
        for batch in dataloader:
            images, captions, masks, cats = batch
            images = images.to(device)
            captions = captions.to(device)
            masks = masks.to(device)
            cats = cats.to(device)
            img_embs,text_embs, logit_scale=model(images,captions,masks)
            loss, uniloss, contrastiveloss = loss_fn(img_embs, text_embs, cats, logit_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_uni += uniloss.item()
            total_contrastive += contrastiveloss.item()

            scheduler.step()
        writer.add_scalar("Loss_train", total_loss / len(dataloader), epoch+1)
        writer.add_scalar("Uni Loss Training", total_uni / len(dataloader), epoch+1)
        writer.add_scalar("Contrastive loss Training", contrastiveloss / len(dataloader), epoch+1)

        print(f"TRAINING: Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(dataloader):.4f}")
        '''# 2. Norma dei gradienti (se sono ~0, il modello non impara)
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item()
        print(f"Gradient Norm: {total_grad_norm}")'''
        
        
        '''state = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'train_loss': total_loss/len(dataloader)
            }
        torch.save(state, f"./checkpoint_train.pth")'''
        validation(dataloader=val_dataloader, model=model, loss_fn=loss_fn,writer=writer, train_epoch=epoch, device=device)

def validation(dataloader, model,loss_fn, writer, train_epoch, device):
    global best_val_loss, best_recall_5_focuss, best_recall_5_all
    model.eval()
    all_img_embs, all_text_embs, all_cats = [], [],[]
    total_loss = 0
    total_uni = 0 
    total_contrastive = 0
    with torch.no_grad():
        for batch in dataloader:
            images, captions, masks, cats = batch
            images = images.to(device)
            captions = captions.to(device)
            masks = masks.to(device)
            cats = cats.to(device)
            img_embs,text_embs, logit_scale=model(images,captions,masks)
            all_img_embs.append(img_embs)
            all_text_embs.append(text_embs)
            all_cats.append(cats)
            loss, uniloss, contrastiveloss = loss_fn(text_embs,img_embs,cats, logit_scale)
            total_loss += loss.item()
            total_uni += uniloss.item()
            total_contrastive += contrastiveloss.item()
    writer.add_scalar("Val_Loss", total_loss / len(dataloader), train_epoch+1)
    writer.add_scalar("Uni Loss Validation", total_uni / len(dataloader), train_epoch+1)
    writer.add_scalar("Contrastive loss Validation", contrastiveloss / len(dataloader), train_epoch+1)

    
    '''if (total_loss/len(dataloader)) < best_val_loss:
        best_val_loss = total_loss/len(dataloader)
        state = {
            'epoch': train_epoch+1,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss
            }
        torch.save(state, f"./best_val_loss.pth")'''
    all_img_embs = torch.cat(all_img_embs, dim=0)
    all_text_embs = torch.cat(all_text_embs, dim=0)
    all_cats = torch.cat(all_cats,dim=0)
    results = compute_metrics(writer=writer,image_embeddings=all_img_embs,epoch=train_epoch, text_embeddings=all_text_embs, categories=all_cats)
    
    '''if (results["exact_focus_R@5"] > best_recall_5_focuss):
        best_recall_5_focuss = results["exact_focus_R@5"]
        state = {
            'epoch': train_epoch+1,
            'state_dict': model.state_dict(),
            'val_loss': total_loss/len(dataloader),
            'best_recall_5': best_recall_5_focuss
            }
        torch.save(state, f"./best_recall@5_focus.pth")
    
    if (results["cat_all_R@5"] > best_recall_5_all):
        best_recall_5_all = results["cat_all_R@5"]
        state = {
            'epoch': train_epoch+1,
            'state_dict': model.state_dict(),
            'val_loss': total_loss/len(dataloader),
            'best_recall_5': best_recall_5_all
            }
        torch.save(state, f"./best_recall@5_all.pth")'''
    
    print(f"VALIDATION = Epoch {train_epoch+1}, Loss: {total_loss/len(dataloader):.4f}, RECALL@5_turtle: {results['exact_focus_R@5']}, RECALL@5_all: {results['cat_all_R@5']}")
    
def compute_metrics(writer, text_embeddings, image_embeddings, epoch,k_values=[1, 5, 10], categories=None):
    global focus_ids
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

    # 2. Calcolo matching esatto solo turtle. 
    focus_mask = torch.isin(categories, torch.tensor(focus_ids, device=categories.device))
    focus_indices = torch.where(focus_mask)[0]  # Indici dei sample con categoria in focus_ids
    if len(focus_indices) > 0:
        # Calcola top-k solo per i sample focus
        sim_focus = sim_matrix[focus_indices]  # [N_focus, N_images]
        top_k = torch.topk(sim_focus, k=max(k_values), dim=1).indices  # [N_focus, max_k]
        for k in k_values:
            # Verifica matching esatto (i-esima query -> i-esima immagine)
            correct = (top_k[:, :k] == focus_indices.unsqueeze(1)).any(dim=1).sum().item()
            recall = correct / len(focus_indices)
            
            results[f"exact_focus_R@{k}"] = recall
            writer.add_scalar(f"exact_turtle_R@{k}", recall, epoch + 1)
    else:
        # Se non ci sono sample focus, imposta recall a 0
        for k in k_values:
            print("\tWARNING: No focus indices in validation found.")
            results[f"exact_focus_R@{k}"] = 0.0
            writer.add_scalar(f"exact_focus_R@{k}", 0.0, epoch + 1)
    
    return results


    return results

    '''
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
        '''



'''def compute_metrics_cpu(writer, text_embeddings, image_embeddings, epoch, k_values=[1, 5, 10], categories=None):
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
    focus_mask = torch.isin(categories, torch.tensor(focus_id))  # Rimossa l'assegnazione al device
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
'''