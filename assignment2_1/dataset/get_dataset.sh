#!/bin/bash

# Exit on error
set -e

# Setup directories
echo "Creating data folder..."
mkdir -p data
cd data

# 1. Download MNIST using a quick torchvision command
echo "Downloading MNIST..."
python3 -c "
import torchvision
torchvision.datasets.MNIST(root='.', train=True, download=True)
torchvision.datasets.MNIST(root='.', train=False, download=True)
"

# 2. Download CelebA files (using official gdown IDs)
echo "Preparing CelebA directory..."
mkdir -p celeba
cd celeba

# Install gdown if not available
pip install -q gdown

echo "Downloading CelebA evaluation partitions..."
gdown --id 0B7EVK8r0v71pY0NSN1gyOTB1ZzQ -O list_eval_partition.txt || echo "Warning: Could not download partitions"
gdown --id 0B7EVK8r0v71pblRyaVlhT1FqOTg -O list_attr_celeba.txt || echo "Warning: Could not download attributes"
gdown --id 0B7EVK8r0v71pY21aZlh5DXNXY28 -O identity_CelebA.txt || echo "Warning: Could not download identities"

echo "Downloading CelebA images (img_align_celeba.zip)..."
# Note: Google Drive quota limits might occasionally apply to this file
gdown --id 0B7EVK8r0v71pTzRFS2t3NjZSMzg -O img_align_celeba.zip

echo "Extracting CelebA images..."
unzip -q img_align_celeba.zip
rm img_align_celeba.zip

echo "Datasets successfully prepared!"
