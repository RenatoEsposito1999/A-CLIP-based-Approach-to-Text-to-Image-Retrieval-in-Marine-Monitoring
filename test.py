from tqdm import tqdm
import torch
import copy
import torch.nn.functional as F
import json
import os
import re

class tester():
    def __init__(self, opts, model, loss_fn, test_loader):
        self.opts = opts
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        last_state = torch.load("/workspace/CLIP/text-to-image-retrivial/best_model_mix.pth")
        self.model.load_state_dict(last_state)
    def test(self):
        self.model.eval()
        all_text_embeds = []
        all_image_embeds = []
        all_categories = []
        total_test_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                images = batch["images"].to(self.opts.device)
                text_inputs = batch["captions"].to(self.opts.device)
                attention_mask = batch['attention_mask'].to(self.opts.device)
                categories = batch["category_id"] #for multiclass contrastive
                image_embeds, text_embeds, logit_scale = self.model(images, text_inputs, attention_mask)
                categories = batch["category_id"]
                test_loss = self.loss_fn(text_embeds, image_embeds, categories ,logit_scale)
                total_test_loss += test_loss.item()
                all_image_embeds.append(image_embeds)
                all_text_embeds.append(text_embeds)
                all_categories.append(categories)
            all_image_embeds = torch.cat(all_image_embeds, dim=0)
            all_text_embeds = torch.cat(all_text_embeds, dim=0)
            all_categories = torch.cat(all_categories, dim=0)
            
            metrics = self.compute_metrics(all_text_embeds,all_image_embeds, all_categories )
            metrics['COCO_TURTLE_test_loss']=total_test_loss/len(self.test_loader)
            # Write everything to JSON file
            with open("./metrics/metrics_test.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4, ensure_ascii=False)
        
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
        ranks_turtle = []

        for i in range(sim_matrix.size(0)):
            sim_scores = sim_matrix[i]
            sorted_indices = torch.argsort(sim_scores, descending=True)

            # Positive = immagini con stessa category_id
            positive_indices = (labels == labels[i]).nonzero(as_tuple=True)[0].to(device)
            found = (sorted_indices.unsqueeze(1) == positive_indices).any(dim=1)
            rank = found.nonzero(as_tuple=True)[0][0].item()
            ranks_all.append(rank)

            # Se la category_id è 'turtle' (0), salva anche nel gruppo dedicato
            if labels[i].item() == 0:
                ranks_turtle.append(rank)

        ranks_all = torch.tensor(ranks_all, device=device)
        ranks_turtle = torch.tensor(ranks_turtle, device=device) if ranks_turtle else torch.tensor([float("inf")], device=device)

        metrics = self.calc_recall(ranks_all,suffix = "COCO_TURTLE_")
        turtle_metrics = self.calc_recall(ranks_turtle,suffix="TURTLE_ONLY_")
        #turtle_metrics = {f"{suffix}turtle_{k}": v for k, v in turtle_metrics.items()}
        metrics.update(turtle_metrics)
        return metrics
        