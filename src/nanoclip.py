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
from src.models import ImageEncoder, TextEncoder


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
        self.loss_fn = ContrastiveLoss(temperature=0.05)

    
    def configure_optimizers(self):
        """
        Define the optimizer and the learning rate scheduler.
        """
        optimizer_params = [
            {"params": self.img_encoder.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": self.txt_encoder.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mult
        )    
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
        self.recall_1_all = 0
        self.recall_5_all = 0
        self_recall_10_all = 0
        self.recall_1_turtle = 0
        self.recall_5_turtle = 0
        self.recall_10_turtle = 0
        self.num_batches = 0
        
    def validation_step(self, batch, batch_idx):
        """ 
        Define a single validation step (one batch pass).
        """
        print("STARTING VALIDATION")
        images, captions, masks, flag = batch
        
        img_descriptors, txt_descriptors = self(images, captions, masks)
        val_loss, val_batch_accuracy = self.loss_fn(img_descriptors, txt_descriptors, flag)
        
        self.log("Val_loss", val_loss, prog_bar=True, logger=True)
        self.log("Val_batch_acc", val_batch_accuracy, prog_bar=True, logger=True)
        
        img_descriptors = img_descriptors.detach().cpu().numpy()
        txt_descriptors = txt_descriptors.detach().cpu().numpy()
        flag = flag.detach().cpu().numpy()
        B = img_descriptors.shape[0]
        labels = np.arange(B)
        recall_list_all, recall_list_turtle = self._calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10], flags=flag)
        recall_1_all, recall_5_all, recall_10_all = recall_list_all
        recall_1_turtle, recall_5_turtle, recall_10_turtle = recall_list_turtle
        self.recall_1_all += recall_1_all
        self.recall_5_all +=recall_5_all
        self.recall_10_all += recall_10_all
        self.recall_1_turtle += recall_1_turtle
        self.recall_5_turtle += recall_5_turtle
        self.recall_10_turtle += recall_5_turtle
        self.num_batches += 1
        
        
        
        
        '''self.validation_descriptors["img"].append(img_descriptors)
        self.validation_descriptors["txt"].append(txt_descriptors)
        self.validation_descriptors["flag"].append(flag)'''
        
    def on_validation_epoch_end(self):
        self.recall_1_all /= self.num_batches
        self.recall_5_all /= self.num_batches
        self.recall_10_all /= self.num_batches
        self.recall_1_turtle /= self.num_batches
        self.recall_5_turtle /= self.num_batches
        self.recall_10_turtle /= self.num_batches
        self.log("all_recall@1", self.recall_1_all, prog_bar=True, logger=True)
        self.log("all_recall@5", self.recall_5_all, prog_bar=True, logger=True)
        self.log("all_recall@10", self.recall_10_all, prog_bar=False, logger=True)
        
        self.log("turtle_recall@1", self.recall_1_turtle, prog_bar=True, logger=True)
        self.log("turtle_recall@5", self.recall_5_turtle, prog_bar=True, logger=True)
        self.log("turtle_recall@10", self.recall_10_turtle, prog_bar=False, logger=True)

    '''
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
        recall_list_all, recall_list_turtle = self._calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10], flags=flag_descriptors)
        recall_1_all, recall_5_all, recall_10_all = recall_list_all
        recall_1_turtle, recall_5_turtle, recall_10_turtle = recall_list_turtle
        self.log("all_recall@1", recall_1_all, prog_bar=True, logger=True)
        self.log("all_recall@5", recall_5_all, prog_bar=True, logger=True)
        self.log("all_recall@10", recall_10_all, prog_bar=False, logger=True)
        
        self.log("turtle_recall@1", recall_1_turtle, prog_bar=True, logger=True)
        self.log("turtle_recall@5", recall_5_turtle, prog_bar=True, logger=True)
        self.log("turtle_recall@10", recall_10_turtle, prog_bar=False, logger=True)

        # clear the validation descriptors for the next epoch
        self.validation_descriptors.clear()
    '''
    
    '''@staticmethod
    def _calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10]):
        """ 
        Calculate the recall at k for the given img_descriptors as gallery
        and txt_descriptors as queries.
        """
        #SE VUOI LA SIMILARITA' L2 DECOMMENTA QUESTE DUE LINEE DI CODICE E COMMENTA LE ALTRE DA faiss.normalize_L2 fino a faiss_index = faiss.IndexFlatIP(embed_size)
        #embed_size = img_descriptors.shape[1]
        #faiss_index = faiss.IndexFlatL2(embed_size)
        
        # Normalize the descriptors to unit length for cosine similarity
        faiss.normalize_L2(img_descriptors)
        faiss.normalize_L2(txt_descriptors)
        
        embed_size = img_descriptors.shape[1]
        # Use IndexFlatIP (Inner Product) for cosine similarity
        faiss_index = faiss.IndexFlatIP(embed_size) 
        
        faiss_index.add(img_descriptors) # add images to the index
        _, predictions = faiss_index.search(txt_descriptors, max(k_values)) # search for the top k images for each text query
        
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], labels[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k /= len(labels)
                
        return correct_at_k'''
        
        
    @staticmethod
    def _calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10], flags=None):
        """ 
        Calculate the recall at k for the given img_descriptors as gallery
        and txt_descriptors as queries.
        """
        #SE VUOI LA SIMILARITA' L2 DECOMMENTA QUESTE DUE LINEE DI CODICE E COMMENTA LE ALTRE DA faiss.normalize_L2 fino a faiss_index = faiss.IndexFlatIP(embed_size)
        #embed_size = img_descriptors.shape[1]
        #faiss_index = faiss.IndexFlatL2(embed_size)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(img_descriptors)
        faiss.normalize_L2(txt_descriptors)

        embed_size = img_descriptors.shape[1]
        faiss_index = faiss.IndexFlatIP(embed_size)
        faiss_index.add(img_descriptors)

        _, predictions = faiss_index.search(txt_descriptors, max(k_values))

        correct_at_k_all = np.zeros(len(k_values))
        correct_at_k_flagged = np.zeros(len(k_values))

        relevant_indices = np.where(flags == -1)[0]

        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                if np.any(np.in1d(pred[:n], labels[q_idx])):
                    correct_at_k_all[i:] += 1
                    if q_idx in relevant_indices:
                        correct_at_k_flagged[i:] += 1
                    break

        correct_at_k_all /= len(labels)
        if len(relevant_indices) > 0:
            correct_at_k_flagged /= len(relevant_indices)
        else:
            correct_at_k_flagged[:] = 0.0

        return correct_at_k_all, correct_at_k_flagged
    
    @staticmethod
    def compute_recall_exact_only(img_descriptors, txt_descriptors, labels,k_values=[1,5,10], flags=None):
        faiss.normalize_L2(img_descriptors)
        faiss.normalize_L2(txt_descriptors)

        embed_size = img_descriptors.shape[1]
        faiss_index = faiss.IndexFlatIP(embed_size)
        faiss_index.add(img_descriptors)

        _, predictions = faiss_index.search(txt_descriptors, max(k_values))

        correct_at_k = np.zeros(len(k_values))
        per_class_recall = {cat: np.zeros(len(k_values)) for cat in [-1, -2, -3, -4, 0]}
        per_class_counts = {cat: 0 for cat in [-1, -2, -3, -4, 0]}

        for q_idx, pred in enumerate(predictions):
            gt_label = labels[q_idx]
            cat_flag = flags[q_idx]
            per_class_counts[cat_flag] += 1

            # Filtra solo se il flag è negativo (-1, -2, -3, -4)
            if cat_flag < 0:
                same_flag_indices = np.where(flags == cat_flag)[0]
                same_flag_indices = same_flag_indices[same_flag_indices != gt_label]  # Escludi il GT stesso
                pred_filtered = [idx for idx in pred if idx not in same_flag_indices]
                pred_filtered += [idx for idx in pred if idx in same_flag_indices]  # Metti in fondo le stesse categorie
            else:
                # Se flag=0, non filtrare nulla
                pred_filtered = pred.copy()

            for i, n in enumerate(k_values):
                if gt_label in pred_filtered[:n]:
                    correct_at_k[i] += 1
                    per_class_recall[cat_flag][i] += 1
                    

        correct_at_k /= len(labels)
        for cat in per_class_recall:
            if per_class_counts[cat] > 0:
                per_class_recall[cat] /= per_class_counts[cat]

        return correct_at_k, per_class_recall
