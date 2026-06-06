# VAE Training, Testing, and Inference 
---

## Setup 

Install packages
```bash
pip install -r requirements.txt
```

---

## 1. Training the VAE

Train the VAE model using  `main.py`.

### Train with Default Config
To train the baseline VAE with hyperparameters loaded from `config.yaml`:
```bash
python main.py --config config.yaml
```

### Train with CLI Parameter Overrides
Override configs directly from the command line:
```bash
# Train on CelebA, using custom data directory and batch configurations
python main.py --dataset celeba --data_dir ./my_celeba_data --epochs 30 --batch_size 128
```

*nho  dang nhap wandb/lay token tu wandb de lay log va bieu do*

---

## 2. Inference & Image Generation

Generate random samples or reconstruct test images using a saved model checkpoint:

### Generate Random Samples (Sample Mode)
Generate new images from the latent space prior and save them as grid comparison files:
```bash
python infer/infer_baseline_vae.py --checkpoint ./checkpoints/best_checkpoint.pth --mode sample --num_samples 16 --output_dir ./results
```

### Reconstruct Dataset Images (Reconstruct Mode)
Load test images, pass them through the VAE, and save side-by-side reconstruction comparisons:
```bash
python infer/infer_baseline_vae.py --checkpoint ./checkpoints/best_checkpoint.pth --mode reconstruct --num_samples 8 --output_dir ./results
```

---

## 3. Running Unit Tests

Run unit tests to verify the pipeline components (datasets, model, and trainer):

### Run All Unit Tests
Run the entire unittest suite:
```bash
python -m unittest discover -s test -p "test_*.py"
```

### Run Specific Unit Tests
To run tests only for specific modules:
```bash
# Test dataset loaders, splits, and tensor shapes
python -m unittest test/test_dataset.py

# Test model architecture layer-by-layer activation shapes
python -m unittest test/test_model.py

# Test training gradients backprop, optimization updates, and log saving
python -m unittest test/test_trainer.py
```
