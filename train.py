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
    def __init__(self,model,loss_fn, optimizer, scheduler, opts):
        self.opts = opts
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.best_val_loss = float('inf')
        self.best_val_loss_turtle = float('inf')
        self.last_epoch = 0
        if self.opts.resume:
            #Retrive last epoch
            checkpoint = find_highest_checkpoint(opts.resume_path)
            if checkpoint:
                print(checkpoint)
                last_state = torch.load(checkpoint)
                self.model.load_state_dict(last_state['state_dict'])
                self.optimizer.load_state_dict(last_state['optimizer'])
                self.scheduler.load_state_dict(last_state['scheduler'])
                self.last_epoch = last_state['epoch']
                self.best_val_loss = last_state['best_val_loss']
                self.best_val_loss_turtle = last_state['best_val_loss_turtle']
                if self.opts.lora:
                    self.model.lora_true()

    def train_loop(self,train_loader, val_loader=None):
        for epoch in tqdm(range(self.last_epoch,self.opts.n_epochs)):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader):
                images = batch["images"].to(self.opts.device)
                text_inputs = batch["captions"].to(self.opts.device)
                attention_mask = batch['attention_mask'].to(self.opts.device)
                categories = batch["category_id"] #for multiclass contrastive
                image_embeds, text_embeds, logit_scale = self.model(images, text_inputs, attention_mask)
                loss = self.loss_fn(text_embeds,image_embeds,categories, logit_scale)

                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                self.scheduler.step()
        
                total_loss += loss.item()
        
            print(f"Epoch {epoch+1}/{self.opts.n_epochs}, Loss: {total_loss/len(train_loader):.4f}")
            # Saving checkpoint
            state = {
                    'epoch': epoch+1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'best_val_loss_turtle': self.best_val_loss_turtle
                    }
            torch.save(state, f"{self.opts.resume_path}checkpoint.pth")

            if val_loader:
                print("START VALIDATION")
                metrics = self.eval_loop(val_loader=val_loader)
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
                if metrics['COCO_TURTLE_val_loss'] < self.best_val_loss:
                    self.best_val_loss = metrics['COCO_TURTLE_val_loss']
                    best_model_state = self.model.state_dict()
                    torch.save(best_model_state, self.opts.best_model_mix_path)
                    print(f"Best model saved at {epoch+1} with val_loss {metrics['COCO_TURTLE_val_loss']:.4f}")
                
    def eval_loop(self, val_loader):
        self.model.eval()
        all_text_embeds = []
        all_image_embeds = []
        all_categories = []
        
        with torch.no_grad():
            total_val_loss = 0
            for batch in tqdm(val_loader):
                images = batch["images"].to(self.opts.device)
                text_inputs = batch["captions"].to(self.opts.device)
                attention_mask = batch['attention_mask'].to(self.opts.device)
                categories = batch["category_id"] #for multiclass contrastive
                image_embeds, text_embeds, logit_scale = self.model(images, text_inputs, attention_mask)
                categories = batch["category_id"]
                val_loss = self.loss_fn(text_embeds, image_embeds, categories ,logit_scale)
                total_val_loss += val_loss.item()

                all_image_embeds.append(image_embeds)
                all_text_embeds.append(text_embeds)
                all_categories.append(categories)

                                

            all_image_embeds = torch.cat(all_image_embeds, dim=0)
            all_text_embeds = torch.cat(all_text_embeds, dim=0)
            all_categories = torch.cat(all_categories, dim=0)
            
            metrics = self.compute_metrics(all_text_embeds,all_image_embeds, all_categories )
            metrics['COCO_TURTLE_val_loss']=total_val_loss/len(val_loader)
            return metrics

    
    def calc_recall(self,ranks,suffix=""):
            return {
                f"{suffix}R@1": (ranks < 1).float().mean().item(),
                f"{suffix}R@5": (ranks < 5).float().mean().item(),
                f"{suffix}R@10": (ranks < 10).float().mean().item(),
                f"{suffix}mean_rank": ranks.float().mean().item(),
            }


    def compute_metrics(self, text_embeds, image_embeds, labels):
        """
        Calcola Recall@K e Mean Rank per:
        - Tutte le categorie
        - Solo categoria 'turtle' (label == 0)
        """
        sim_matrix = text_embeds @ image_embeds.T  # [N_text, N_image]
        device = sim_matrix.device

        ranks_all = []
        
        unique_id_coco = 1
        category_id = []
        for id in labels:
            if id == 2:
                category_id.append(-2) #turtle
            elif id == 5:
                category_id.append(-5) #sea
            elif id == 10:
                category_id.append(-10) #dolphin
            elif id == 11:
                category_id.append(-11) #debris
            else:
                category_id.append(unique_id_coco)
                unique_id_coco += 1
        labels = torch.tensor(category_id)
       

        '''for i in range(sim_matrix.size(0)):
            sim_scores = sim_matrix[i]
            sorted_indices = torch.argsort(sim_scores, descending=True)

            # Positive = immagini con stessa category_id
            positive_indices = (labels == labels[i]).nonzero(as_tuple=True)[0].to(device)
            
            found = (sorted_indices.unsqueeze(1) == positive_indices).any(dim=1)
            rank = found.nonzero(as_tuple=True)[0][0].item()
            ranks_all.append(rank)

            # Se la category_id è 'turtle' (0), salva anche nel gruppo dedicato
            if labels[i].item() == -2:
                ranks_turtle.append(rank)'''
        # One TURTLE/sea/delphine/debris VS OTHERS 
        for i in range(sim_matrix.size(0)):
            sim_scores = sim_matrix[i]
            sorted_indices = torch.argsort(sim_scores, descending=True)

            # Ottieni label della caption i
            label_i = labels[i].item()

            # Trova immagine corretta (è l'indice stesso: i <-> i)
            true_index = i

            # Escludi immagini con stessa categoria (tranne la vera immagine)
            mask = torch.ones_like(labels, dtype=torch.bool, device=device)  # Inizia con tutti True
            if label_i < 0:  # categorie "da ignorare", tipo -2 ("turtle"), -5, ecc.
                same_category = (labels == label_i).to(device)
                mask = ~same_category  # Escludi tutte le immagini con la stessa categoria
                mask[true_index] = True  # Riabilita l'immagine corretta
            mask = mask.to(device)
            # Applica la maschera ai punteggi di similarità
            filtered_sim_scores = sim_scores.masked_fill(~mask, float('-inf'))

            # Ordina le immagini per similarità
            sorted_indices = torch.argsort(filtered_sim_scores, descending=True)

            # Trova il rank della vera immagine
            rank = (sorted_indices == true_index).nonzero(as_tuple=True)[0][0].item()
            ranks_all.append(rank)



        ranks_all = torch.tensor(ranks_all, device=device)

        metrics = self.calc_recall(ranks_all,suffix = "COCO_TURTLE_")
        return metrics
