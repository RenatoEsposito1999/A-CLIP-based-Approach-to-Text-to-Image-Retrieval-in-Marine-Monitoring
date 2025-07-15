import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
import torch.nn.functional as F
import csv
import os
focus_ids = [-2]

class Tester:
    def __init__(self, model, dataloader, loss, device, model_name):
        self.dataloader = dataloader
        self.loss_fn = loss
        self.model_name = model_name
        state = torch.load("best_recall@5_focus.pth")
        model.load_state_dict(state["state_dict"])
        self.model = model.to(device)
        self.device = device
    def test(self):
        self.model.eval()
        all_img_embs, all_text_embs, all_cats = [], [],[]
        total_loss = 0
        total_uni = 0 
        total_contrastive = 0
        with torch.no_grad():
            for batch in self.dataloader:
                images, captions, masks, cats = batch
                images = images.to(self.device)
                captions = captions.to(self.device)
                masks = masks.to(self.device)
                cats = cats.to(self.device)
                img_embs,text_embs, logit_scale=self.model(images,captions,masks)
                all_img_embs.append(img_embs)
                all_text_embs.append(text_embs)
                all_cats.append(cats)
                loss, uniloss, contrastiveloss = self.loss_fn(text_embs,img_embs,cats, logit_scale)
                total_loss += loss.item()
                total_uni += uniloss.item()
                total_contrastive += contrastiveloss.item()
        all_img_embs = torch.cat(all_img_embs, dim=0)
        all_text_embs = torch.cat(all_text_embs, dim=0)
        all_cats = torch.cat(all_cats,dim=0)
        test_file = open("./result_test.csv", mode='a', encoding='utf-8', newline='')
        fieldnames = ['model_name', 'test_contrastive_loss', 'test_uni_loss', 'test_loss']
        for k in [1,5,10]:
            fieldnames.append(f"cat_all_R@{k}")
        for k in [1,5,10]:
            fieldnames.append(f"exact_focus_R@{k}")
        writer_test = csv.DictWriter(test_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if not os.path.isfile("./result_test.csv"):
            writer_test.writeheader()
        results = self.compute_metrics(text_embeddings=all_text_embs, image_embeddings=all_img_embs, categories=all_cats)
        row = {
            "model_name": self.model_name,
            "test_contrastive_loss": total_contrastive / len(self.dataloader),
            "test_uni_loss": total_uni / len(self.dataloader),
            "test_loss": total_loss / len(self.dataloader),
        } | results
        
        print(row)
        writer_test.writerow(row)
        
        
    def compute_metrics(self, text_embeddings, image_embeddings, k_values=[1, 5, 10], categories=None):
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
            #writer.add_scalar(f"cat_all_R@{k}", results[f"cat_all_R@{k}"], epoch+1)

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
                #writer.add_scalar(f"exact_turtle_R@{k}", recall, epoch + 1)
        else:
            # Se non ci sono sample focus, imposta recall a 0
            for k in k_values:
                print("\tWARNING: No focus indices in validation found.")
                results[f"exact_focus_R@{k}"] = 0.0
                #writer.add_scalar(f"exact_focus_R@{k}", 0.0, epoch + 1)
    
        return results