import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from transformers import AutoProcessor
from src.sampler import NonRepeatingBalancedSampler
from src.dataset import dataset, Collate_fn
from src.model import CLIP_model
from src.loss import compute_loss
from src.trainer import Trainer
from src.tester import Tester
from utils.seed import seed_everything
from utils.token import CHAT_ID_RENATO, CHAT_ID_VINCENZO
from utils.telegram_notification import send_telegram_notification
from utils.get_optimizer_and_scheduler import get_optimizer_and_scheduler
from utils.version_log_tensorboard import get_next_version


def main(batch_size, lr, device, wd, n_epochs, no_train : bool, test : bool, model_name: str, resume=False, checkpoint=None):
    seed = 12345
    seed_everything(seed)
    
       
    if device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
  
    #DEFINE THE MODEL CLIP
    model = CLIP_model(model_name=model_name)
    
    #processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    processor = AutoProcessor.from_pretrained(model_name)
    collate_fn = Collate_fn(processor=processor)
    
    
    

    if not no_train:
        #PREPARING TENSOBOARD
        log_base_dir = "logs/CLIP"
        next_version = get_next_version(log_base_dir)
        log_dir = os.path.join(log_base_dir, f"version_{next_version}")
        writer = SummaryWriter(log_dir=log_dir)
        # print numbers of params of the model     
        total_params = sum(p.numel() for p in model.parameters())    
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}\nTotal parameters: {total_params:,}")
        

        print("Train dataset")
        print("-"*15)
        train_dataset = dataset("./datasets/", split="train", seed=seed)
        print("-"*15)
        print("Validation dataset")
        val_dataset = dataset("./datasets/", split="val", seed=seed)
        print("-"*15)
        
        train_sampler = NonRepeatingBalancedSampler(dataset=train_dataset, batch_size=batch_size, fixed_categories=[-2])
        val_sampler = NonRepeatingBalancedSampler(dataset=val_dataset, batch_size=batch_size, fixed_categories=[-2])
        train_dataloader = DataLoader(
                            train_dataset, 
                            batch_sampler = train_sampler, 
                            num_workers=4, 
                            pin_memory=True, 
                            collate_fn=collate_fn 
                        )
        val_dataloader = DataLoader(
                            val_dataset,
                            batch_sampler = val_sampler,
                            num_workers=4, 
                            pin_memory=True,
                            collate_fn=collate_fn
                        )
        
        print("Start training")
        optimizer, scheduler = get_optimizer_and_scheduler(model, lr=lr,weight_decay=wd, tot_num_epochs=n_epochs, steps_per_epoch=len(train_dataloader))
        trainer = Trainer(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, loss=compute_loss, optimizer=optimizer, scheduler=scheduler, writer_log=writer, device=device, n_epoch=n_epochs, resume=resume, checkpoint=checkpoint)
        send_telegram_notification(message="Inizio il Training!", CHAT_ID=[CHAT_ID_RENATO,CHAT_ID_VINCENZO])
        trainer.fit()
        send_telegram_notification(message="Training completato!", CHAT_ID=[CHAT_ID_RENATO,CHAT_ID_VINCENZO])

    if test:
        print("Test dataset")
        test_dataset = dataset("./datasets/", split="test", seed=seed)
        print("-"*15)
        test_sampler = NonRepeatingBalancedSampler(dataset=test_dataset, batch_size=batch_size, fixed_categories=[-2])
        test_dataloader = DataLoader(
                            test_dataset, 
                            batch_sampler = test_sampler,
                            num_workers=4, 
                            pin_memory=True,
                            collate_fn=collate_fn
                        )
        
        tester = Tester(model=model, dataloader=test_dataloader, loss=compute_loss, device=device, model_name="CLIP_OpenAI_v3")
        tester.test()

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train parameters")
    
    parser.add_argument("--dev", action="store_true", help="Enable fast dev run (one train and validation iteration).")
    parser.add_argument("--bs", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning Rate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--wd", type=float, default=4e-4, help="Weight decay")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epoch")
    parser.add_argument("--no_train", type=bool, default=False, help="True if want to NO TRAIN")
    parser.add_argument("--test", type=bool, default=True, help="True if want to do TEST")
    #laion/CLIP-ViT-B-32-laion2B-s34B-b79K
    #openai/clip-vit-base-patch32
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="Pretrained model name")
    parser.add_argument("--resume", type=bool, default=False, help="Boolean value if want to resume")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path for resuming the training")
    args = parser.parse_args()
    main(batch_size=args.bs, 
         lr=args.lr, 
         device= args.device, 
         wd = args.wd, 
         n_epochs=args.n_epochs, 
         no_train=args.no_train, 
         test=args.test, 
         model_name=args.model_name, 
         resume=args.resume, 
         checkpoint=args.checkpoint_path)