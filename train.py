from tqdm import tqdm
import torch
import copy
import torch.nn.functional as F
import json
import os
import re
def find_highest_checkpoint(resume_path):
    max_num = -1
    pattern = re.compile(r'checkpoint_(\d+)\.pth')  # Regex per estrarre i numeri
    
    for filename in os.listdir(resume_path):
        match = pattern.match(filename)
        if match:
            current_num = int(match.group(1))
            if current_num > max_num:
                max_num = current_num
    
    if max_num == -1:
        return None  # Nessun checkpoint trovato
    else:
        return resume_path+f"checkpoint_{max_num}.pth"


class Train:
    def __init__(self,model,loss_fn, optimizer, opts):
        self.opts = opts
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model
        self.best_val_loss = float('inf')
        self.last_epoch = 0
        if self.opts.resume:
            #Retrive last epoch
            checkpoint = find_highest_checkpoint(opts.resume_path)
            if checkpoint:
                last_state = torch.load(checkpoint)
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
                categories = batch["category_id"] #for multiclass contrastive
                image_embeds, text_embeds = self.model(images, text_inputs)
                #loss = self.loss_fn(image_embeds, text_embeds, self.model.logit_scale, turtle)
                loss = self.loss_fn(text_embeds,image_embeds,categories, self.model.logit_scale)

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
            torch.save(state, f"{self.opts.resume_path}checkpoint_{epoch+1}.pth")

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
        all_categories = []
        with torch.no_grad():
            total_val_loss = 0
            for batch in tqdm(val_loader):
                images = batch["images"].to(self.opts.device)
                text_inputs = {k: v.to(self.opts.device) for k, v in batch["captions"].items()}
                image_embeds, text_embeds = self.model(images, text_inputs)
                categories = batch["category_id"]
                #val_loss = self.loss_fn(image_embeds, text_embeds, self.model.logit_scale, categories)
                val_loss = self.loss_fn(text_embeds, image_embeds, categories ,self.model.logit_scale)
                total_val_loss += val_loss.item()

                all_image_embeds.append(image_embeds)
                all_text_embeds.append(text_embeds)
                all_categories.append(categories)

            all_image_embeds = torch.cat(all_image_embeds, dim=0)
            all_text_embeds = torch.cat(all_text_embeds, dim=0)
            all_categories = torch.cat(all_categories, dim=0)
            if is_only_turtle:
                metrics = self.compute_metrics(all_text_embeds,all_image_embeds, all_categories, suffix = "only_turtle_")
                metrics['only_turtle_val_loss']=total_val_loss/len(val_loader)
            else:
                metrics = self.compute_metrics(all_text_embeds,all_image_embeds, all_categories, suffix="COCO_TURTLE_")
                metrics['val_loss']=total_val_loss/len(val_loader)
            return metrics
            #print(f"Validation loss: {total_val_loss/len(val_loader):.4f}\nMetrics: {metrics}")

    def compute_metrics(self,text_embeds, image_embeds, labels, suffix=""):
        """
        image_embeds: [N, D] - image features
        text_embeds: [N, D] - text features
        labels: [N] - category ID, shared between image[i] and text[i]
        """
        # Calcolo similarità (dot product o cosine se già normalizzati)
        sim_matrix = text_embeds @ image_embeds.T  # [N_text, N_image]

        ranks = []
        for i in range(sim_matrix.size(0)):
            sim_scores = sim_matrix[i]  # similarità tra caption i e tutte le immagini
            sorted_indices = torch.argsort(sim_scores, descending=True)

            # Indici delle immagini con stessa categoria della caption i
            positive_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]

            # Trova la prima immagine positiva nella lista ordinata
            found = (sorted_indices.unsqueeze(1) == positive_indices).any(dim=1)
            rank = found.nonzero(as_tuple=True)[0][0]  # indice in cui ho trovato il primo match
            ranks.append(rank.item())

        # Metriche
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