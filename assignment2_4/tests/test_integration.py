import os
import sys
import yaml
import pytest
import importlib.util

# Mock torchao check to bypass version incompatibility in peft under older Colab environments
_original_find_spec = importlib.util.find_spec
def _mocked_find_spec(name, package=None):
    if name == "torchao" or name.startswith("torchao."):
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _mocked_find_spec

def test_configs_exist():
    config_path = "dreamboth-lora-trainer/training_config.yaml"
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert "model" in config
    assert "data" in config
    assert "training" in config
    assert "logging" in config

def test_imports():
    # Verify we can import everything from our source code
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dreamboth-lora-trainer")))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dreamboth-lora-trainer/src")))
    
    from src.dataset import DreamBoothDataset, DreamBoothCollator, PromptDataset
    from src.model_utils import load_base_models, inject_lora, save_lora_weights
    from src.pipeline import TrainingPipeline
    
    assert True
