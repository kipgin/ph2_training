import os
import sys
import yaml
import json
import torch
import argparse
import wandb
import torch.optim as optim
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

# Add root folder of assignment2_2 to system path for clean imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet
from models.ddpm import DDPM
from dataset.datasets import MNISTDataset, CelebADataset

class DDPMTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        #setup dataset& loaders
        self.dataset_name = config['dataset'].lower()
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.image_size = config['image_size']
        self.num_workers = config['num_workers']
        
        if self.dataset_name == 'mnist':
            self.dataset_wrapper = MNISTDataset(
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                val_split_ratio=config['val_split_ratio'],
                num_workers=self.num_workers
            )
        elif self.dataset_name == 'celeba':
            self.dataset_wrapper = CelebADataset(
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                val_split_ratio=config['val_split_ratio'],
                num_workers=self.num_workers
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        self.train_loader, self.val_loader, self.test_loader = self.dataset_wrapper.get_loaders()
        
        #setup UNet& DDPM wrapper models
        in_channels = self.dataset_wrapper.in_channels
        self.config['model']['in_channels'] = in_channels
        
        self.unet = UNet(
            in_channels=in_channels,
            time_emb_dim=self.config['model']['time_emb_dim'],
            hidden_dims=self.config['model']['hidden_dims']
        )
        
        self.model = DDPM(
            unet=self.unet,
            num_timesteps=self.config['diffusion']['num_timesteps'],
            beta_start=self.config['diffusion']['beta_start'],
            beta_end=self.config['diffusion']['beta_end'],
            schedule_name=self.config['diffusion']['schedule_name']
        ).to(self.device)
        
        # 3. Setup optimizer and scheduler
        lr = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['training']['epochs'], eta_min=1e-6
        )
        
        # 4. Setup output directories
        self.checkpoint_dir = self.config['training']['checkpoint_dir']
        self.samples_dir = os.path.join(self.checkpoint_dir, 'samples')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # 5. Initialize tracking history logs
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'fid_samples': []
        }
        self.best_val_loss = float('inf')
        
        # 6. Initialize Weights & Biases (wandb) logging
        print("Initializing Weights & Biases (wandb) run...")
        self.run = wandb.init(
            project=self.config['training']['wandb_project'],
            name=f"ddpm_{self.dataset_name}_{self.image_size}x{self.image_size}",
            config=self.config
        )
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.config['training']['epochs']}]")
        for batch_idx, (imgs, _) in enumerate(loop):
            imgs = imgs.to(self.device)
            
            self.optimizer.zero_grad()
            
            #forward
            loss = self.model(imgs)
            
            #backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.5f}")
            
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
        
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for imgs, _ in self.val_loader:
                imgs = imgs.to(self.device)
                
                loss = self.model(imgs)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
        
    def save_samples(self, epoch):
        self.model.eval()
        num_samples = 16
        in_channels = self.dataset_wrapper.in_channels
        shape = (num_samples, in_channels, self.image_size, self.image_size)
        
        print(f"--> Generating {num_samples} samples via reverse DDPM loop...")
        with torch.no_grad():
            samples = self.model.p_sample_loop(shape, device=self.device)
            #normalize [-1, 1] back to [0, 1]
            samples = (samples + 1.0) / 2.0
            
            grid = make_grid(samples, nrow=4)
            output_path = os.path.join(self.samples_dir, f"sample_epoch_{epoch+1}.png")
            save_image(grid, output_path)
            print(f"--> Saved samples to {output_path}")
            
            # Log grid image directly to wandb
            self.run.log({
                "val/generation_grid": wandb.Image(grid, caption=f"Random Samples Epoch {epoch+1}")
            })
            
    def run_eval_metrics(self, epoch):
        num_samples = self.config['training']['num_fid_samples']
        print(f"--> Running FID evaluation (samples: {num_samples})...")
        try:
            fid_score = self.dataset_wrapper.calculate_fid(
                self.model, self.device, num_samples=num_samples, mode='sample'
            )
            print(f"    [FID Score]: {fid_score:.4f}")
            self.history['fid_samples'].append(fid_score)
            
            #log to wandb
            self.run.log({"metrics/fid": fid_score})
        except Exception as e:
            print(f"    [FID Error]: Could not calculate FID score: {e}")
            self.history['fid_samples'].append(None)
            
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"New best validation loss achieved. Saved best checkpoint to {best_path}")
            
    def save_history(self):
        history_path = os.path.join(self.checkpoint_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
            
    def train(self):
        epochs = self.config['training']['epochs']
        sample_interval = self.config['training']['sample_interval']
        fid_interval = self.config['training']['fid_interval']
        
        print("Starting DDPM Training Loop...")
        for epoch in range(epochs):
            # train epoch
            train_loss = self.train_epoch(epoch)
            
            #validate
            val_loss = self.validate()
            
            #step scheduler
            self.scheduler.step()
            
            print(f"Summary - Epoch [{epoch+1}/{epochs}]")
            print(f"    Train Loss: {train_loss:.5f}")
            print(f"    Val Loss:   {val_loss:.5f}")
            
            #log history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            #log to wandb
            self.run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            #save samples
            if (epoch + 1) % sample_interval == 0:
                self.save_samples(epoch)
                
            #run FID score evaluation
            if (epoch + 1) % fid_interval == 0:
                self.run_eval_metrics(epoch)
            else:
                self.history['fid_samples'].append(None)
                
            #save Checkpoints
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            self.save_checkpoint(is_best=is_best)
            self.save_history()
            
        #close wandb run
        self.run.finish()

def main():
    parser = argparse.ArgumentParser(description="Trainer script for DDPM")
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser.add_argument('--config', type=str, default=os.path.join(root_path, 'config.yaml'), help='Path to YAML config file')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'celeba'], help='Override dataset selection')
    parser.add_argument('--data_dir', type=str, help='Override dataset directory')
    parser.add_argument('--epochs', type=int, help='Override training epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.dataset:
        config['dataset'] = args.dataset
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
        
    print("------- Configuration -------")
    print(yaml.dump(config))
    print("-----------------------------")
    
    trainer = DDPMTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
