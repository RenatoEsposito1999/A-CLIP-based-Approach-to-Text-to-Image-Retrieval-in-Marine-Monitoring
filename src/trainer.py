import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
import torch.nn.functional as F


class Trainer():
    """
        A complete training pipeline for contrastive learning models with text-image retrieval capabilities.

        Key Functionality:
        - Manages full training lifecycle with progress tracking
        - Computes Recall@k metrics during validation
        - Maintains multiple checkpoint strategies
        - Provides comprehensive logging via TensorBoard

        Attributes:
            model (torch.nn.Module): The neural network being trained
            device (str): Computation device ('cpu' or 'cuda')
            focus_ids (list): Special category IDs for focused evaluation (default [-2])
            best_val_loss (float): Tracked best validation loss
            best_recall_5_focus (float): Best focus category Recall@5 score
            best_recall_5_all (float): Best overall Recall@5 score

        Methods:
            __init__(): Initializes trainer with model, data loaders, and training components
            train_one_epoch(): Executes a single training epoch
            validation(): Runs full validation cycle
            compute_metrics(): Calculates retrieval metrics (Recall@k)
            train(): Main training loop that orchestrates the process

        Example:
            >>> trainer = Trainer(model, train_loader, val_loader, loss_fn,
            ...                 optimizer, scheduler, writer, device='cuda')
            >>> trainer.fit()
    """

    def __init__(self, model, train_dataloader, val_dataloader, loss, optimizer, scheduler, writer_log, device="cpu", n_epoch=50, resume=False, checkpoint=None):
        """
            Initializes the model trainer with training/validation components and optional resume capability.
            
            Args:
                model: The neural network model to train
                train_dataloader: DataLoader for training data
                val_dataloader: DataLoader for validation data
                loss: Loss function module
                optimizer: Optimization algorithm
                scheduler: Learning rate scheduler
                writer_log: TensorBoard writer for logging
                device: Target device for training ('cpu' or 'cuda')
                n_epoch: Total number of training epochs
                resume: Whether to resume from checkpoint
                checkpoint: Path to checkpoint file when resuming
                
            Initializes:
                - Model and data loaders on specified device
                - Training components (optimizer, loss, scheduler)
                - Tracking variables for best metrics
                - Focus category IDs for special evaluation
                - Either fresh training state or restored checkpoint state
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
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
            
    def fit(self):
        """
            Main training loop with validation and model checkpointing.
            Per epoch:
            1. Trains model for one epoch and logs training losses
            2. Runs validation and computes retrieval metrics
            3. Tracks and logs:
            - All training/validation losses (total, uniform, contrastive)
            - Recall@k metrics (both category-all and focus-category)
            4. Maintains three separate model checkpoints:
            - Best validation loss (lowest)
            - Best focus category Recall@5 (highest)
            - Best overall Recall@5 (highest)
            5. Saves full training state (model+optimizer) every epoch

            Uses tqdm for progress tracking and TensorBoard writer for logging.
        """
        for epoch in tqdm(range(self.start_epoch, self.n_epoch)):
            self.model.train()
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
            self.writer_log.add_scalar(f"cat_all_mean_rank", metric_results["cat_all_mean_rank"], epoch + 1)
            self.writer_log.add_scalar(f"exact_focus_mean_rank", metric_results["exact_focus_mean_rank"], epoch + 1)
            print(f"VALIDATION = Epoch {epoch+1}, Loss: {val_loss:.4f}, RECALL@5_turtle: {metric_results['exact_focus_R@5']}, RECALL@5_all: {metric_results['cat_all_R@5']}")
            
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
        """
            Executes one training epoch for the contrastive learning model.
            
            Processes batches to:
            - Compute image/text embeddings
            - Calculate joint loss (combination of contrastive and uniform loss)
            - Perform backpropagation and optimization
            
            Returns:
                tuple: (avg_total_loss, avg_uniform_loss, avg_contrastive_loss) 
                    averaged across all batches
        """
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
            total_loss += loss.item()
            total_uni_loss += uniloss.item()
            total_contrastive_loss += contrastiveloss.item()
        return total_loss/len(self.train_dataloader), total_uni_loss/len(self.train_dataloader), total_contrastive_loss/len(self.train_dataloader)

    def validation(self):
        """
            Computes text-to-image retrieval metrics:
            - Global Recall@k for all categories
            - Exact matching Recall@k for specific focus categories
            
            Args:
                text_embeddings (torch.Tensor): Text embeddings [N_text, dim]
                image_embeddings (torch.Tensor): Image embeddings [N_images, dim]
                k_values (list): k values for Recall@k (default [1, 5, 10])
                categories (torch.Tensor): Category labels for all samples [N_total]
                
            Returns:
                dict: Dictionary with computed metrics
        """
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
                
    '''def compute_metrics(self, text_embeddings, image_embeddings, k_values=[1, 5, 10], categories=None):
        text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_embeddings_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        sim_matrix = torch.mm(text_embeddings_norm, image_embeddings_norm.t())  #[N_text, N_images]
    
        # Get the top-k (descending) indices
        top_k_indices = torch.topk(sim_matrix, k=max(k_values), dim=1).indices  # [N_text, max(k_values)]

        results = {}
        n_total = len(text_embeddings)
    
        # 1. GLOBAL calculation (entire dataset)
        for k in k_values:
            # Get the categories retrieved for each query
            retrieved_cats = categories[top_k_indices[:, :k]]  # [N_text, k]
            query_cats = categories.unsqueeze(1)  # [N_text, 1]
        
            # Check if the query category is among the retrieve ones
            correct = (retrieved_cats == query_cats).any(dim=1).sum().item()
            results[f"cat_all_R@{k}"] = correct / n_total
            #writer.add_scalar(f"cat_all_R@{k}", results[f"cat_all_R@{k}"], epoch+1)

        # 2. Exact matching calculation turtle only.
        focus_mask = torch.isin(categories, torch.tensor(self.focus_ids, device=categories.device))
        focus_indices = torch.where(focus_mask)[0]  # Indici dei sample con categoria in focus_ids
        if len(focus_indices) > 0:
            # Calculate top-k only for focus samples
            sim_focus = sim_matrix[focus_indices]  # [N_focus, N_images]

            


            top_k = torch.topk(sim_focus, k=max(k_values), dim=1).indices  # [N_focus, max_k]
            for k in k_values:
                # Check exact matching (i-th query -> i-th image)
                correct = (top_k[:, :k] == focus_indices.unsqueeze(1)).any(dim=1).sum().item()
                recall = correct / len(focus_indices)
                results[f"exact_focus_R@{k}"] = recall
        else:
            # If there is no sample focus, set recall to 0
            for k in k_values:
                print("\tWARNING: No focus indices in validation found.")
                results[f"exact_focus_R@{k}"] = 0.0
        return results'''


    def compute_metrics(self, text_embeddings, image_embeddings, k_values=[1, 5, 10], categories=None):
        text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_embeddings_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    
        sim_matrix = torch.mm(text_embeddings_norm, image_embeddings_norm.t())  #[N_text, N_images]
    
        # Ottieni gli indici top-k (discendenti)
        top_k_indices = torch.topk(sim_matrix, k=max(k_values), dim=1).indices  # [N_text, max(k_values)]

        results = {}
        n_total = len(text_embeddings)
    
        # 1. Calcolo GLOBALE (tutto il dataset)
        # Calculate sorted indices for all samples (for mean rank)
        sorted_indices_all = torch.argsort(sim_matrix, dim=1, descending=True)  # [N_text, N_images]
        # Calculate mean rank for category matching
        category_ranks = []


        for i in range(n_total):
            query_cat = categories[i]
            # Find the first occurrence of the correct category in sorted results
            ranked_cats = categories[sorted_indices_all[i]]  # categories in order of similarity
            rank = (ranked_cats == query_cat).nonzero()[0].item() + 1  # +1 because rank starts at 1
            category_ranks.append(rank)
        results["cat_all_mean_rank"] = sum(category_ranks) / n_total


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


            # Calculate mean rank for exact matching
            sorted_indices = torch.argsort(sim_focus, dim=1, descending=True)  # [N_focus, N_images]
            
            ranks = []
            for i, idx in enumerate(focus_indices):
                rank = (sorted_indices[i] == idx).nonzero().item() + 1  # +1 because rank starts at 1
                ranks.append(rank)
            mean_rank = sum(ranks) / len(ranks)
            results["exact_focus_mean_rank"] = mean_rank



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