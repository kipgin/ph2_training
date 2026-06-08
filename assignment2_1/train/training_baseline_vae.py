import os
import sys
import yaml
import json
import torch
import argparse
import wandb
import torch.optim as optim
from torch.nn import functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


from models.baseline_vae import BaselineVAE
from dataset.datasets import MNISTDataset, CelebADataset

class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
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
        
        in_channels = self.dataset_wrapper.in_channels
        self.config['base_vae']['in_channels'] = in_channels
        
        self.model = BaselineVAE(
            in_channels=in_channels,
            hidden_dims=self.config['base_vae']['hidden_dims'],
            latent_dim=self.config['base_vae']['latent_dim'],
            kld_weight=self.config['base_vae']['kld_weight']
        ).to(self.device)
        
        
        lr = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        

        self.checkpoint_dir = self.config['training']['checkpoint_dir']
        self.samples_dir = os.path.join(self.checkpoint_dir, 'samples')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        

        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kld': [],
            'val_loss': [],
            'val_recon': [],
            'val_kld': [],
            'fid_samples': [],
            'fid_recons': []
        }
        self.best_val_loss = float('inf')
        
        
        print("Initializing Weights & Biases (wandb) run...")
        self.run = wandb.init(
            project="VAE-Baseline",
            name=f"baseline_vae_{self.dataset_name}_{self.config['base_vae']['latent_dim']}d",
            config=self.config
        )
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.config['training']['epochs']}]")
        for batch_idx, (imgs, _) in enumerate(loop):
            imgs = imgs.to(self.device)
            batch_size = imgs.size(0)
            
            self.optimizer.zero_grad()
            
            x_recon, mu, logvar = self.model(imgs)
            
            recon_loss = F.mse_loss(x_recon, imgs, reduction='sum') / batch_size
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            loss = recon_loss + kld_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            
            loop.set_postfix(
                loss=f"{loss.item():.4f}", 
                recon=f"{recon_loss.item():.4f}", 
                kld=f"{kld_loss.item():.4f}"
            )
            
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kld = total_kld / num_batches
        
        return avg_loss, avg_recon, avg_kld
        
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        with torch.no_grad():
            for imgs, _ in self.val_loader:
                imgs = imgs.to(self.device)
                batch_size = imgs.size(0)
                
                x_recon, mu, logvar = self.model(imgs)
                
                recon_loss = F.mse_loss(x_recon, imgs, reduction='sum') / batch_size
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
                loss = recon_loss + kld_loss
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld_loss.item()
                
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kld = total_kld / num_batches
        
        return avg_loss, avg_recon, avg_kld
        
    def save_samples(self, epoch):
        self.model.eval()
        with torch.no_grad():
            
            real_batch, _ = next(iter(self.val_loader))
            real_batch = real_batch[:8].to(self.device)
            recon_batch, _, _ = self.model(real_batch)
            
            comparison = torch.cat([real_batch, recon_batch], dim=0)
            comparison = (comparison + 1.0) / 2.0  # normalize back to [0, 1]
            grid_recon = make_grid(comparison, nrow=8)
            save_image(grid_recon, os.path.join(self.samples_dir, f"recon_epoch_{epoch+1}.png"))
            
            num_samples = 16
            samples = self.model.sample(num_samples, current_device=self.device)
            samples = (samples + 1.0) / 2.0  # normalize back to [0, 1]
            grid_sample = make_grid(samples, nrow=4)
            save_image(grid_sample, os.path.join(self.samples_dir, f"sample_epoch_{epoch+1}.png"))
            
            print(f"--> Saved samples & reconstructions to {self.samples_dir}")
            
            self.run.log({
                "val/reconstruction_grid": wandb.Image(grid_recon, caption=f"Reconstruction Comparison Epoch {epoch+1}"),
                "val/generation_grid": wandb.Image(grid_sample, caption=f"Random Samples Epoch {epoch+1}")
            })

    def run_eval_metrics(self, epoch):
        num_samples = self.config['training']['num_fid_samples']
        print(f"--> Running FID evaluation (mode: sample, samples: {num_samples})...")
        try:
            fid_sample = self.dataset_wrapper.calculate_fid(self.model, self.device, num_samples=num_samples, mode='sample')
            print(f"    [FID Generation]: {fid_sample:.4f}")
            self.history['fid_samples'].append(fid_sample)
            # Log metric to wandb
            self.run.log({"metrics/fid_generation": fid_sample})
        except Exception as e:
            print(f"    [FID Generation Error]: Could not calculate FID score: {e}")
            self.history['fid_samples'].append(None)
            
        try:
            fid_recon = self.dataset_wrapper.calculate_fid(self.model, self.device, num_samples=num_samples, mode='reconstruct')
            print(f"    [FID Reconstruction]: {fid_recon:.4f}")
            self.history['fid_recons'].append(fid_recon)
            # Log metric to wandb
            self.run.log({"metrics/fid_reconstruction": fid_recon})
        except Exception as e:
            print(f"    [FID Reconstruction Error]: Could not calculate FID score: {e}")
            self.history['fid_recons'].append(None)
            
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        # Save latest model checkpoint
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
        
        print("Starting VAE Training Loop...")
        for epoch in range(epochs):
            train_loss, train_recon, train_kld = self.train_epoch(epoch)
        
            val_loss, val_recon, val_kld = self.validate()
            
            #update lr
            self.scheduler.step(val_loss)
            
            #print epoch summary
            print(f"Summary - Epoch [{epoch+1}/{epochs}]")
            print(f"    Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KLD: {train_kld:.4f})")
            print(f"    Val Loss:   {val_loss:.4f} (Recon: {val_recon:.4f}, KLD: {val_kld:.4f})")
            
            #update history log
            self.history['train_loss'].append(train_loss)
            self.history['train_recon'].append(train_recon)
            self.history['train_kld'].append(train_kld)
            self.history['val_loss'].append(val_loss)
            self.history['val_recon'].append(val_recon)
            self.history['val_kld'].append(val_kld)
            
            #log metrics to wandb
            self.run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/recon_loss": train_recon,
                "train/kld_loss": train_kld,
                "val/loss": val_loss,
                "val/recon_loss": val_recon,
                "val/kld_loss": val_kld,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            #save reconstruction & random samples
            if (epoch + 1) % sample_interval == 0:
                self.save_samples(epoch)
                
            #run FID score evaluation
            if (epoch + 1) % fid_interval == 0:
                self.run_eval_metrics(epoch)
            else:
                #add None or dummy placeholder to keep history list aligned with epochs
                self.history['fid_samples'].append(None)
                self.history['fid_recons'].append(None)
                
            #checkpoint management
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            self.save_checkpoint(is_best=is_best)
            self.save_history()
            
        # Finish the wandb logging session
        self.run.finish()

def main():
    parser = argparse.ArgumentParser(description="Trainer script for Baseline VAE")
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser.add_argument('--config', type=str, default=os.path.join(root_path, 'config.yaml'), help='Path to YAML configuration file')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'celeba'], help='Override dataset selection')
    parser.add_argument('--data_dir', type=str, help='Override dataset directory')
    parser.add_argument('--epochs', type=int, help='Override training epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override settings if command line options are provided
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
    
    trainer = VAETrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
