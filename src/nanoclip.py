# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import faiss
import numpy as np
import lightning as L
#from src.loss import ContrastiveLoss
from src.loss_multi_positive_turtle import ContrastiveLoss
from src.supervised_contrastive_loss import SupervisedContrastiveLoss
from src.models import ImageEncoder, TextEncoder
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class NanoCLIP(L.LightningModule):
    """ 
    This class defines the pipeline for the nanoCLIP model.
    
    """
    def __init__(
        self,
        txt_model="sentence-transformers/all-MiniLM-L6-v2",
        img_model='dinov2_vits14',
        embed_size=64, # output dimension of the encoder
        unfreeze_n_blocks=4,
        lr=0.0001,
        warmup_epochs=0,
        weight_decay=0.0001,
        milestones=[5, 10, 15],
        lr_mult=0.1,
    ):
        super().__init__()
        
        self.txt_model = txt_model
        self.img_model = img_model
        self.embed_size = embed_size
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.lr_mult = lr_mult
        
        self.save_hyperparameters() # save all hyperparameters to hparams file (for reproducibility) 
        
        self.img_encoder = ImageEncoder(self.embed_size, self.img_model, unfreeze_n_blocks)
        self.txt_encoder = TextEncoder(self.embed_size, self.txt_model, unfreeze_n_blocks)
        self.loss_fn = SupervisedContrastiveLoss(temperature=0.05)
        self.focus_id = []
        with open("/workspace/text-to-image-retrivial/datasets/annotations/category_info.json", "r") as f:
            data = json.load(f)
            self.focus_id.append(data["turtle"][0])
            self.focus_id.append(data["dolphin"][0])
            self.focus_id.append(data["sea"][0])
            self.focus_id.append(data["debris"][0])
        
    
    def configure_optimizers(self):
        """
        Define the optimizer and the learning rate scheduler.
        """
        optimizer_params = [
            {"params": self.img_encoder.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": self.txt_encoder.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)    
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Define how a single optimization step is executed.
        """
        if self.trainer.current_epoch < self.warmup_epochs:
            total_warmup_steps = self.warmup_epochs * self.trainer.num_training_batches
            lr_scale = min(1.0, (self.trainer.global_step + 1) / total_warmup_steps)
            for pg in optimizer.param_groups:
                initial_lr = pg.get("initial_lr", self.lr)
                pg["lr"] = lr_scale * initial_lr

        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)
    
    def forward(self, image, captions, masks):
        """ 
        Define the forward pass of the pipeline.
        """
        # compute image embeddings
        image_embedding = self.img_encoder(image) # (batch_size, out_dim)
        image_embedding = F.normalize(image_embedding, p=2, dim=-1) # normalize embeddings
        
        # compute text embeddings
        text_embedding = self.txt_encoder(captions, masks) # (batch_size, nb_captions, out_dim)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1) # normalize embeddings
        
        return image_embedding, text_embedding
    
    def training_step(self, batch, batch_idx):
        """ 
        Define a single training step (one batch pass).
        
        ImageEncoder ──┐
                       ├──► ContrastiveLoss   
        TextEncoder  ──┘
        """
        images, captions, masks, flag = batch
 
        if len(captions.shape) == 3: # flatten captions to (batch_size*nb_caps, cap_len) cuz we have multiple captions per image
            B, nb_captions, cap_len = captions.shape
            B, nb_masks, mask_len = masks.shape
            captions = captions.view(B*nb_captions, cap_len) 
            masks = masks.view(B*nb_masks, mask_len)
        else:
            nb_captions = 1
            
        img_descriptors, txt_descriptors = self(images, captions, masks)
        
        
        if nb_captions > 1: # reshape back to (B, nb_captions, out_dim)
            txt_descriptors = txt_descriptors.view(B, nb_captions, -1)
        
        
        loss, batch_accuracy = self.loss_fn(img_descriptors, txt_descriptors, flag)
        
        
        self.log("Train_loss", loss, prog_bar=True, logger=True)
        self.log("Train_batch_acc", batch_accuracy, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_start(self):
        self.validation_descriptors = {"img": [], "txt": [], "flag": []}
        
        
    def validation_step(self, batch, batch_idx):
        """ 
        Define a single validation step (one batch pass).
        """
        images, captions, masks, flag = batch
        
        img_descriptors, txt_descriptors = self(images, captions, masks)
        val_loss, val_batch_accuracy = self.loss_fn(img_descriptors, txt_descriptors, flag)
        
        self.log("Val_loss", val_loss, prog_bar=True, logger=True)
        self.log("Val_batch_acc", val_batch_accuracy, prog_bar=True, logger=True)
        
        img_descriptors = img_descriptors.detach().cpu().numpy()
        txt_descriptors = txt_descriptors.detach().cpu().numpy()
        flag = flag.detach().cpu().numpy()
        
        
        self.validation_descriptors["img"].append(img_descriptors)
        self.validation_descriptors["txt"].append(txt_descriptors)
        self.validation_descriptors["flag"].append(flag)
        
    

    
    def on_validation_epoch_end(self):
        """ 
        Calculate the recall at 1, 5, and 10 for the validation set.
        """
        img_descriptors = np.concatenate(self.validation_descriptors["img"], axis=0) # (N, out_dim)
        txt_descriptors = np.concatenate(self.validation_descriptors["txt"], axis=0) # (N, out_dim)
        flag_descriptors = np.concatenate(self.validation_descriptors["flag"], axis=0)
        
        # create dummy labels
        B = img_descriptors.shape[0]    
        labels = np.arange(B)

        # use faiss to calculate recall, images are gallery and texts are queries
        #recall_1, recall_5, recall_10 = self._calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10])
        #recall_list_all, recall_list_turtle = self._calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10], flags=flag_descriptors)
        
        
        # MATTCHING ESATTO DA METTERE IN FUTURO SE HA SENSO 
        '''recall_exactly_matching = self.compute_text_to_image_recall_exactly_matching(img_descriptors, txt_descriptors, k_values=[1, 5, 10], categories=flag_descriptors)
        recall_1_all_exactly_matching = recall_exactly_matching["recall@1"]
        recall_5_all_exactly_matching = recall_exactly_matching["recall@5"]
        recall_10_all_exactly_matching = recall_exactly_matching["recall@10"]
        recall_1_turtle_exactly_matching = recall_exactly_matching["recall@1_turtle"]
        recall_5_turtle_exactly_matching = recall_exactly_matching["recall@5_turtle"]
        recall_10_turtle_exactly_matching = recall_exactly_matching["recall@10_turtle"]
        self.log("exactly_all_r@1", recall_1_all_exactly_matching, prog_bar=True, logger=True)
        self.log("exactly_all_r@5", recall_5_all_exactly_matching, prog_bar=True, logger=True)
        self.log("exactly_all_r@10", recall_10_all_exactly_matching, prog_bar=False, logger=True)
        
        self.log("exactly_turtle_r@1", recall_1_turtle_exactly_matching, prog_bar=True, logger=True)
        self.log("exactly_turtle_r@5", recall_5_turtle_exactly_matching, prog_bar=True, logger=True)
        self.log("exactly_turtle_r@10", recall_10_turtle_exactly_matching, prog_bar=False, logger=True)'''
        
        recall_category_matching = self.compute_recall_per_category_loose(img_descriptors, txt_descriptors, k_values=[1, 5, 10], categories=flag_descriptors)
        recall_1_category = recall_category_matching["cat_all_R@1"]
        recall_5_category = recall_category_matching["cat_all_R@5"]
        recall_10_category = recall_category_matching["cat_all_R@10"]
        recall_1_turtle = recall_category_matching["cat_focus_R@1"]
        recall_5_turtle = recall_category_matching["cat_focus_R@5"]
        recall_10_turtle = recall_category_matching["cat_focus_R@10"]
        
        self.log("category_all_r@1", recall_1_category, prog_bar=True, logger=True)
        self.log("category_all_r@5", recall_5_category, prog_bar=True, logger=True)
        self.log("category_all_r@10", recall_10_category, prog_bar=False, logger=True)
        
        self.log("category_focus_r@1", recall_1_turtle, prog_bar=True, logger=True)
        self.log("category_focus_r@5", recall_5_turtle, prog_bar=True, logger=True)
        self.log("category_focus_r@10", recall_10_turtle, prog_bar=False, logger=True)
        

        # clear the validation descriptors for the next epoch
        self.validation_descriptors.clear()
    
    '''
    @staticmethod
    def _calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10], flag=None):
        """ 
        Calculate the recall at k for the given img_descriptors as gallery
        and txt_descriptors as queries.
        """
        #SE VUOI LA SIMILARITA' L2 DECOMMENTA QUESTE DUE LINEE DI CODICE E COMMENTA LE ALTRE DA faiss.normalize_L2 fino a faiss_index = faiss.IndexFlatIP(embed_size)
        #embed_size = img_descriptors.shape[1]
        #faiss_index = faiss.IndexFlatL2(embed_size)
        
        #flag = [12,2,0,3,5,12]
        
        
        # Normalize the descriptors to unit length for cosine similarity
        faiss.normalize_L2(img_descriptors)
        faiss.normalize_L2(txt_descriptors)
        
        embed_size = img_descriptors.shape[1]
        # Use IndexFlatIP (Inner Product) for cosine similarity
        faiss_index = faiss.IndexFlatIP(embed_size) 
        
        faiss_index.add(img_descriptors) # add images to the index
        _, predictions = faiss_index.search(txt_descriptors, max(k_values)) # search for the top k images for each text query
        #predictions = [1,5,4,3]
        correct_at_k = np.zeros(len(k_values)) #[0,0,0]
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], labels[q_idx])):
                    correct_at_k[i:] += 1
                    #Se category è turtle
                    break
        
        correct_at_k /= len(labels)
                
        return correct_at_k
    '''

    def compute_text_to_image_recall_exactly_matching(self, text_embeddings, image_embeddings, k_values=[1, 5, 10], categories = None):
        """
        Calcola Recall@K per un problema di text-to-image retrieval.
        - Calcolo globale su tutto il dataset
        - Supporta estrazione per singola categoria senza ricalcolare la similarità

        Args:
            text_embeddings (np.ndarray): shape (N, D)
            image_embeddings (np.ndarray): shape (N, D)
            categories (List[str]): categoria di ciascuna coppia
            k_values (List[int]): valori di k per cui calcolare la recall
            category_filter (str, optional): se specificato, calcola solo sulle entry di quella categoria

        Returns:
            dict: recall@k globali e opzionalmente per categoria
        """
        text_embeddings = np.array(text_embeddings)
        image_embeddings = np.array(image_embeddings)
        categories = np.array(categories)
    
        assert text_embeddings.shape[0] == image_embeddings.shape[0] == len(categories)
    
        n = len(text_embeddings)
        sim_matrix = cosine_similarity(text_embeddings, image_embeddings)
        top_k_indices = np.argsort(-sim_matrix, axis=1)  # descending order

        results = {}
        
        # Calcolo recall@k globali
        for k in k_values:
            correct = [i in top_k_indices[i, :k] for i in range(n)]
            results[f"recall@{k}"] = np.mean(correct)

        # Calcolo recall@k per categoria specifica (usando stessi indici)
            mask = np.isin(categories, self.focus_id)
            indices = np.where(mask)[0]
            for k in k_values:
                correct = [i in top_k_indices[i, :k] for i in indices]
                results[f"recall@{k}_turtle"] = np.mean(correct) if indices.size > 0 else 0.0

        return results
    
    '''def compute_recall_per_category_loose(self, text_embeddings, image_embeddings, k_values=[1, 5, 10], categories=None):
        """
        Calcola Recall@K per ogni categoria:
        - Una retrieval è corretta se almeno un'immagine nelle top-k ha la stessa categoria del testo
        - Include macro-media delle recall per k

        Returns:
            dict: {"recall@1_turtle": ..., ..., "recall@1_macro": ...}
        """
        text_embeddings = np.array(text_embeddings)
        image_embeddings = np.array(image_embeddings)
        categories = np.array(categories)

        assert text_embeddings.shape[0] == image_embeddings.shape[0] == len(categories)

        sim_matrix = cosine_similarity(text_embeddings, image_embeddings)
        top_k_indices = np.argsort(-sim_matrix, axis=1)

        results = {}
        unique_categories = np.unique(categories)

        per_k_macro = {k: [] for k in k_values}  # accumula per macro-media

        for cat in unique_categories:
            text_indices = np.where(categories == cat)[0]
            if len(text_indices) == 0:
                continue

            for k in k_values:
                correct = 0
                for i in text_indices:
                    if cat in self.focus_id:
                        results[f"recall@{k}_turtle"] = 0
                    else:
                        results[f"recall@{k}_{cat}"] = 0
                    
                    retrieved_image_indices = top_k_indices[i, :k]
                    retrieved_categories = categories[retrieved_image_indices]
                    if cat in retrieved_categories:
                        correct += 1
                recall = correct / len(text_indices)
                if cat in self.focus_id:
                    results[f"recall@{k}_turtle"] += recall
                else:
                    results[f"recall@{k}_{cat}"] = recall
                per_k_macro[k].append(recall)

        # Calcolo macro-media per ogni k
        for k in k_values:
            if per_k_macro[k]:
                results[f"recall@{k}_macro"] = np.mean(per_k_macro[k])
            else:
                results[f"recall@{k}_macro"] = 0.0
            results[f"recall@{k}_turtle"] /= 4

        return results'''



    def compute_recall_per_category_loose(self, text_embeddings, image_embeddings, k_values=[1, 5, 10], categories=None):
        """
        Calcola:
        - Recall@K globale: almeno un'immagine top-k ha stessa categoria del testo (tutto il dataset)
        - Recall@K focus: come sopra ma solo per categorie in focus_id
        """
        text_embeddings = np.array(text_embeddings)
        image_embeddings = np.array(image_embeddings)
        categories = np.array(categories)
        
        assert text_embeddings.shape[0] == image_embeddings.shape[0] == len(categories)
        
        sim_matrix = cosine_similarity(text_embeddings, image_embeddings)
        top_k_indices = np.argsort(-sim_matrix, axis=1)  # descending order
        results = {}
        
        # 1. Calcolo GLOBALE (tutto il dataset)
        n_total = len(text_embeddings)
        for k in k_values:
            correct = 0
            for i in range(n_total):
                query_cat = categories[i]
                retrieved_cats = categories[top_k_indices[i, :k]]
                if query_cat in retrieved_cats:
                    correct += 1
            results[f"cat_all_R@{k}"] = correct / n_total
        
        # 2. Calcolo FOCUS (solo categorie specificate)
        if hasattr(self, 'focus_id'):
            focus_mask = np.isin(categories, self.focus_id)
            focus_indices = np.where(focus_mask)[0]
            n_focus = len(focus_indices)
            
            for k in k_values:
                correct = 0
                for i in focus_indices:
                    query_cat = categories[i]
                    retrieved_cats = categories[top_k_indices[i, :k]]
                    if query_cat in retrieved_cats:
                        correct += 1
                results[f"cat_focus_R@{k}"] = correct / n_focus if n_focus > 0 else 0.0
        
        return results