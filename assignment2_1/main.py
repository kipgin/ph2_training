import os
import sys
import yaml
import argparse


from train.training_baseline_vae import VAETrainer

def main():
    parser = argparse.ArgumentParser(description="VAE Training Framework CLI Entrypoint")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration YAML file')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'celeba'], help='Override dataset selection (mnist or celeba)')
    parser.add_argument('--data_dir', type=str, help='Override dataset directory path')
    parser.add_argument('--epochs', type=int, help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override training learning rate')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Override training device')
    
    args = parser.parse_args()
    
    # Check and locate configuration file path
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), args.config)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {args.config}")
            
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Apply command-line overrides to YAML configs if provided
    if args.dataset:
        config['dataset'] = args.dataset
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.device:
        config['training']['device'] = args.device
        
    print("======= Training Config Override =======")
    print(yaml.dump(config))
    print("========================================")
    
    trainer = VAETrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
