# DDPM and DDIM Diffusion Framework

A clean, modular, and robust PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM), complete with U-Net architecture, MNIST and CelebA dataset wrappers, FID metric calculations, and Weights & Biases (`wandb`) logging.

---

## Setup & Dependencies

Install all project dependencies:
```bash
pip install -r requirements.txt
```

---

## 1. Training the Diffusion Model

Train the diffusion model using the CLI entrypoint `main.py`.

### Train with Default Config
To train DDPM using parameters loaded from `config.yaml` (default to MNIST at 32x32 resolution):
```bash
python main.py --config config.yaml
```

### Train with CLI Overrides
Override configs directly from the command line:
```bash
# Train on CelebA, using custom data directory and batch configurations
python main.py --dataset celeba --data_dir ./my_celeba_data --epochs 60 --batch_size 64
```

*Note: Make sure you log into your W&B account via `wandb login` before starting training, or set `WANDB_MODE=offline` to track training metrics locally.*

---

## 2. Inference & Image Generation (DDPM vs. DDIM)

Generate new samples using a saved checkpoint file. The inference script `infer/samplers.py` supports two samplers:

### Standard Stochastic Sampling (DDPM Mode)
Runs the complete stochastic reverse process over all $T$ steps (e.g. 1000 steps):
```bash
python infer/samplers.py --checkpoint ./checkpoints/best_checkpoint.pth --sampler ddpm --num_samples 16 --output_dir ./results
```

### Accelerated Deterministic Sampling (DDIM Mode)
Accelerates inference by skipping steps (e.g., executing in only 50 steps instead of 1000):
```bash
python infer/samplers.py --checkpoint ./checkpoints/best_checkpoint.pth --sampler ddim --ddim_steps 50 --num_samples 16 --output_dir ./results
```

---

## 3. Running Unit Tests

Run unit tests to verify the datasets, U-Net, and DDPM training loop gradient flow:

### Run All Unit Tests
```bash
python -m unittest discover -s test -p "test_*.py"
```

### Run Specific Unit Tests
To run tests only for specific modules:
```bash
# Test dataset loaders, splits, and tensor shapes
python -m unittest test/test_dataset.py

# Test U-Net shape changes and DDPM forward/reverse loop consistencies
python -m unittest test/test_model.py

# Test training gradients flow, optimization updates, and log saving
python -m unittest test/test_trainer.py
```
