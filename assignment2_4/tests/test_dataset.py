import os
import shutil
import tempfile
import numpy as np
from PIL import Image
import torch
import pytest
import sys
import importlib.util

# Mock torchao check to bypass version incompatibility in peft under older Colab environments
_original_find_spec = importlib.util.find_spec
def _mocked_find_spec(name, package=None):
    if name == "torchao" or name.startswith("torchao."):
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _mocked_find_spec

# Add the dreamboth-lora-trainer directory to the python path for importing src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dreamboth-lora-trainer")))

from src.dataset import DreamBoothDataset, PromptDataset, DreamBoothCollator

class MockTokenizer:
    def __init__(self):
        self.model_max_length = 10

    def __call__(self, text, padding=None, truncation=None, max_length=None):
        class Output(dict):
            def __init__(self):
                super().__init__()
                self.input_ids = [1, 2, 3]
                self["input_ids"] = self.input_ids
        return Output()

    def pad(self, dict_of_ids, padding=None, max_length=None, return_tensors=None):
        ids = dict_of_ids["input_ids"]
        padded = []
        for seq in ids:
            padded_seq = seq + [0] * (max_length - len(seq))
            padded.append(padded_seq)
        class Output(dict):
            def __init__(self, input_ids):
                super().__init__()
                self.input_ids = input_ids
                self["input_ids"] = input_ids
        return Output(torch.tensor(padded))

def test_dreamboth_dataset():
    # Create temp directories for instance and class images
    with tempfile.TemporaryDirectory() as tmpdir:
        instance_dir = os.path.join(tmpdir, "instance")
        class_dir = os.path.join(tmpdir, "class")
        os.makedirs(instance_dir)
        os.makedirs(class_dir)

        # Save dummy images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(instance_dir, f"instance_{i}.png"))
        
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(class_dir, f"class_{i}.png"))

        tokenizer = MockTokenizer()
        
        # Test without class images (no prior preservation)
        dataset_no_prior = DreamBoothDataset(
            instance_data_root=instance_dir,
            instance_prompt="a photo of a cool cat",
            tokenizer=tokenizer,
            size=64,
            center_crop=True
        )

        assert len(dataset_no_prior) == 3
        item = dataset_no_prior[0]
        assert "instance_images" in item
        assert "instance_prompt_ids" in item
        assert "class_images" not in item
        assert item["instance_images"].shape == (3, 64, 64)

        # Test with class images (with prior preservation)
        dataset_with_prior = DreamBoothDataset(
            instance_data_root=instance_dir,
            instance_prompt="a photo of a cool cat",
            tokenizer=tokenizer,
            class_data_root=class_dir,
            class_prompt="a photo of a cat",
            size=64,
            center_crop=True
        )

        assert len(dataset_with_prior) == 5
        item = dataset_with_prior[0]
        assert "instance_images" in item
        assert "instance_prompt_ids" in item
        assert "class_images" in item
        assert "class_prompt_ids" in item
        assert item["instance_images"].shape == (3, 64, 64)
        assert item["class_images"].shape == (3, 64, 64)


def test_dreamboth_collator():
    with tempfile.TemporaryDirectory() as tmpdir:
        instance_dir = os.path.join(tmpdir, "instance")
        class_dir = os.path.join(tmpdir, "class")
        os.makedirs(instance_dir)
        os.makedirs(class_dir)

        # Save dummy images
        for i in range(2):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(instance_dir, f"instance_{i}.png"))
        
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(class_dir, f"class_{i}.png"))

        tokenizer = MockTokenizer()
        
        # Test without prior preservation
        dataset = DreamBoothDataset(
            instance_data_root=instance_dir,
            instance_prompt="a photo of a cool cat",
            tokenizer=tokenizer,
            size=64
        )
        collator = DreamBoothCollator(tokenizer=tokenizer, with_prior_preservation=False)
        
        batch = collator([dataset[0], dataset[1]])
        assert batch["pixel_values"].shape == (2, 3, 64, 64)
        assert batch["input_ids"].shape == (2, 10)

        # Test with prior preservation
        dataset_prior = DreamBoothDataset(
            instance_data_root=instance_dir,
            instance_prompt="a photo of a cool cat",
            tokenizer=tokenizer,
            class_data_root=class_dir,
            class_prompt="a photo of a cat",
            size=64
        )
        collator_prior = DreamBoothCollator(tokenizer=tokenizer, with_prior_preservation=True)
        
        batch_prior = collator_prior([dataset_prior[0], dataset_prior[1]])
        # Since with_prior_preservation=True, batch size is doubled: 2 * instance + 2 * class = 4
        assert batch_prior["pixel_values"].shape == (4, 3, 64, 64)
        assert batch_prior["input_ids"].shape == (4, 10)


def test_prompt_dataset():
    prompt_ds = PromptDataset("a picture of a tree", 10)
    assert len(prompt_ds) == 10
    item = prompt_ds[4]
    assert item["prompt"] == "a picture of a tree"
    assert item["index"] == 4
