import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
import torch.nn.functional as F

import torch.nn.functional as F


class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader, loss, optimizer, scheduler, writer_log, device="cpu", n_epoch=50, resume=False, checkpoint=None):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader,
        self.loss_fn = loss
        self.device = device
        self.n_epoch = n_epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer_log = writer_log
        self.focus_ids = [-2]
    
        if resume:
            checkpoint = torch.load(checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = checkpoint["epoch"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.best_val_loss = checkpoint["best_val_loss"]
            self.best_recall_5_focus = checkpoint["best_recall_5_focus"]
            self.best_recall_5_all = checkpoint["best_recall_5_all"]
        else:
            self.start_epoch = 0
            self.best_val_loss = float("inf")
            self.best_recall_5_focus = -float("inf")
            self.best_recall_5_all = -float("inf")
            
    def train(self):
        for epoch in tqdm(range(self.start_epoch, self.n_epoch)):
            self.model.train()
            self.train_one_epoch()
            train_loss, train_uni_loss, train_contrastive_loss = self.train_one_epoch()
            self.writer_log.add_scalar("Loss_train", train_loss, epoch+1)
            self.writer_log.add_scalar("Uni Loss Training", train_uni_loss, epoch+1)
            self.writer_log.add_scalar("Contrastive loss Training", train_contrastive_loss, epoch+1)
            print(f"TRAINING: Epoch {epoch+1}/{self.n_epoch}, Loss: {train_loss:.4f}")
            
            val_loss, val_uni_loss, val_contrastive_loss, metric_results =self.validation()
            
            self.writer_log.add_scalar("Val_Loss", val_loss, epoch+1)
            self.writer_log.add_scalar("Uni Loss Validation", val_uni_loss, epoch+1)
            self.writer_log.add_scalar("Contrastive loss Validation", val_contrastive_loss, epoch+1)
            
            for k in [1,5,10]:
                self.writer_log.add_scalar(f"cat_all_R@{k}", metric_results[f"cat_all_R@{k}"], epoch+1)
                self.writer_log.add_scalar(f"exact_turtle_R@{k}", metric_results[f"exact_focus_R@{k}"], epoch + 1)
            print(f"""VALIDATION = Epoch {epoch+1}, 
                  Loss: {val_loss:.4f}, 
                  RECALL@5_turtle: {metric_results['exact_focus_R@5']}, 
                  RECALL@5_all: {metric_results['cat_all_R@5']}""")
            
            if (val_loss < self.best_val_loss):
                self.best_val_loss = val_loss
                state = {
                    'epoch': epoch+1,
                    'state_dict': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss
                    }
                torch.save(state, f"./best_val_loss.pth")
            
            if (metric_results["exact_focus_R@5"] > self.best_recall_5_focus):
                self.best_recall_5_focus = metric_results["exact_focus_R@5"]
                state = {
                    'epoch': epoch+1,
                    'state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'best_recall_5': self.best_recall_5_focus
                    }
                torch.save(state, f"./best_recall@5_focus.pth")
    
            if (metric_results["cat_all_R@5"] > self.best_recall_5_all):
                self.best_recall_5_all = metric_results["cat_all_R@5"]
                state = {
                    'epoch': epoch+1,
                    'state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'best_recall_5': self.best_recall_5_all
                    }
                torch.save(state, f"./best_recall@5_all.pth")
                
            state = {
                'epoch': epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': train_loss,
                'best_val_loss': self.best_val_loss,
                'best_recall_5_focus': self.best_recall_5_focus,
                'best_recall_5_all': self.best_recall_5_all,          
                }
            torch.save(state, f"./checkpoint_train.pth")
            

    def train_one_epoch(self):
        total_loss = 0
        total_contrastive_loss = 0
        total_uni_loss = 0
        for batch in self.train_dataloader:
            images, captions, masks, cats = batch
            images = images.to(self.device)
            captions = captions.to(self.device)
            masks = masks.to(self.device)
            cats = cats.to(self.device)
            img_embs,text_embs, logit_scale=self.model(images,captions,masks)
            loss, uniloss, contrastiveloss = self.loss_fn(img_embs, text_embs, cats, logit_scale)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss
            total_uni_loss += uniloss
            total_contrastive_loss += contrastiveloss
        return total_loss/len(self.train_dataloader), total_uni_loss/len(self.train_dataloader), total_contrastive_loss/len(self.train_dataloader)


    def validation(self):
        self.model.eval()
        all_img_embs, all_text_embs, all_cats = [], [],[]
        total_loss = 0
        total_uni = 0 
        total_contrastive = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
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
        results = self.compute_metrics(text_embeddings=all_text_embs, image_embeddings=all_img_embs, categories=all_cats)
        return total_loss / len(self.val_dataloader), total_uni / len(self.val_dataloader), total_contrastive/len(self.val_dataloader), results
                
    def compute_metrics(self, text_embeddings, image_embeddings, k_values=[1, 5, 10], categories=None):
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
        focus_mask = torch.isin(categories, torch.tensor(self.focus_ids, device=categories.device))
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