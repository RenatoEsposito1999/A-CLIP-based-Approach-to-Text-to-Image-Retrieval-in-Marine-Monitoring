from tqdm import tqdm
import torch
import copy
import torch.nn.functional as F
import json
import os
class Train:
    def __init__(self,model,loss_fn, optimizer, opts):
        self.opts = opts
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model
        self.best_val_loss = float('inf')
        self.last_epoch = 0
        if self.opts.resume:
            last_state = torch.load(opts.resume_path+"checkpoint.pth")
            self.model.load_state_dict(last_state['state_dict'])
            self.optimizer.load_state_dict(last_state['optimizer'])
            self.last_epoch = last_state['epoch']
            self.best_val_loss = last_state['best_val_loss']

    def train_loop(self,train_loader, val_loader=None, only_turtle_loader=None):
        for epoch in tqdm(range(self.last_epoch,self.opts.n_epochs)):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader):
                images = batch["images"].to(self.opts.device)
                text_inputs = {k: v.to(self.opts.device) for k, v in batch["captions"].items()}
                turtle = batch["turtle"] #for multiclass contrastive
                image_embeds, text_embeds = self.model(images, text_inputs)
                loss = self.loss_fn(image_embeds, text_embeds, self.model.logit_scale, turtle)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                total_loss += loss.item()
        
            print(f"Epoch {epoch+1}/{self.opts.n_epochs}, Loss: {total_loss/len(train_loader):.4f}")
            # Saving checkpoint
            state = {
                    'epoch': epoch+1,
                    'state_dict': copy.deepcopy(self.model.state_dict()),
                    'optimizer': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss
                    }
            torch.save(state, f"{self.opts.resume_path}checkpoint.pth")

            if val_loader:
                print("START VALIDATION ON COCO AND TURTLE (MIXED) DATASET")
                metrics = self.eval_loop(val_loader=val_loader, is_only_turtle=False)
                if only_turtle_loader:
                    print("START VALIDATION ON ONLY TURTLE DATASET")
                    metrics_only_turtle = self.eval_loop(val_loader=only_turtle_loader,is_only_turtle=True)
                    metrics = metrics | metrics_only_turtle # Merge dicts and overload same key
                metrics['train_loss'] = total_loss/len(train_loader)
                metrics['epoch'] = epoch+1
                print(metrics)
                #Save metrics into a json
                file_json = self.opts.metrics_path+"metrics.json"

                # If file exists, read and load existing data
                if os.path.exists(file_json):
                    with open(file_json, "r", encoding="utf-8") as f:
                        # Load as list or dictionaries
                        existing_data = json.load(f)  
                else:
                    # If it doesn't exist, start with an empty list.
                    existing_data = []  

                # Add new dictionary to list
                existing_data.append(metrics)

                # Write everything to JSON file
                with open(file_json, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=4, ensure_ascii=False)

                # Save best model
                if metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = metrics['val_loss']
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    torch.save(best_model_state, self.opts.best_model_path)
                    print(f"Best model saved at {epoch+1} with val_loss {metrics['val_loss']:.4f}")

    def eval_loop(self, val_loader, is_only_turtle = False):
        self.model.eval()
        all_text_embeds = []
        all_image_embeds = []
        all_turtles = []
        with torch.no_grad():
            total_val_loss = 0
            for batch in tqdm(val_loader):
                images = batch["images"].to(self.opts.device)
                text_inputs = {k: v.to(self.opts.device) for k, v in batch["captions"].items()}
                image_embeds, text_embeds = self.model(images, text_inputs)
                turtle = batch["turtle"]
                val_loss = self.loss_fn(image_embeds, text_embeds, self.model.logit_scale, turtle)
                total_val_loss += val_loss.item()

                all_image_embeds.append(image_embeds)
                all_text_embeds.append(text_embeds)
                all_turtles.append(turtle)

            all_image_embeds = torch.cat(all_image_embeds, dim=0)
            all_text_embeds = torch.cat(all_text_embeds, dim=0)
            all_turtles = torch.cat(all_turtles, dim=0)
            if is_only_turtle:
                metrics = self.compute_metrics(all_image_embeds, all_text_embeds, suffix = "only_turtle_", all_turtles=all_turtles)
                metrics['only_turtle_val_loss']=total_val_loss/len(val_loader)
            else:
                metrics = self.compute_metrics(all_image_embeds, all_text_embeds, suffix="COCO_TURTLE_", all_turtles=all_turtles)
                metrics['val_loss']=total_val_loss/len(val_loader)
            return metrics
            #print(f"Validation loss: {total_val_loss/len(val_loader):.4f}\nMetrics: {metrics}")


    '''def compute_metrics(self,image_embeds, text_embeds, suffix=""):
        # Normalize embeddings (cosine similarity)
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Similarity matrix (text → image)
        sim_matrix = text_embeds @ image_embeds.T  # shape (N_text, N_image)

        # Ground-truth: ogni text_embed[i] ha come immagine corretta image_embed[i]
        ranks = []
        for i in range(sim_matrix.size(0)):

            sim_scores = sim_matrix[i]
            # Ordina immagini da più simile a meno simile per il testo i
            sorted_indices = torch.argsort(sim_scores, descending=True)
            # Trova il rank dell'immagine corretta (che è in posizione i)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)

        ranks = torch.tensor(ranks)
        r1 = (ranks < 1).float().mean().item()
        r5 = (ranks < 5).float().mean().item()
        r10 = (ranks < 10).float().mean().item()
        mean_rank = ranks.float().mean().item()

        return {
            f"{suffix}R@1": r1,
            f"{suffix}R@5": r5,
            f"{suffix}R@10": r10,
            f"{suffix}mean_rank": mean_rank
        }'''
        
    def compute_metrics(self,image_embeds, text_embeds, suffix="", all_turtles=None):
        # Normalize embeddings (cosine similarity)
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        

        sim_matrix = text_embeds @ image_embeds.T  # shape (N_text, N_image)
        # Assumiamo turtle è già tensor: shape (N,), valori 0 (coco) o 1 (tartaruga)
        all_turtles = all_turtles.to(sim_matrix.device)  # Sposta su GPU se necessario
        ranks = []
        for i in range(sim_matrix.size(0)):
            sim_scores = sim_matrix[i]
            sorted_indices = torch.argsort(sim_scores, descending=True)

            if all_turtles[i] == 1:
                # Caption = tartaruga → tutte le immagini tartaruga sono positive
                positive_indices = (all_turtles == 1).nonzero(as_tuple=True)[0]
            else:
                # Caption COCO → solo immagine i è positiva
                positive_indices = torch.tensor([i], device=sim_matrix.device)

            found = (sorted_indices.unsqueeze(1) == positive_indices).any(dim=1)
            rank = found.nonzero(as_tuple=True)[0][0]         
            ranks.append(rank.item())

        ranks = torch.tensor(ranks, device=sim_matrix.device)
        r1 = (ranks < 1).float().mean().item()
        r5 = (ranks < 5).float().mean().item()
        r10 = (ranks < 10).float().mean().item()
        mean_rank = ranks.float().mean().item()

        return {
            f"{suffix}R@1": r1,
            f"{suffix}R@5": r5,
            f"{suffix}R@10": r10,
            f"{suffix}mean_rank": mean_rank
        }