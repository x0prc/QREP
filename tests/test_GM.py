"""
Unit tests for GANManager and StyleGANTrainer.
Tests:
1. Checkpoint saving and loading.
2. Sample image and metadata saving.
"""

import os
import shutil
import torch
import json
import pytest
from src.differential.gan_manager import GANManager

@pytest.fixture(scope="function")
def temp_dirs():
    models_dir = "test_gan_models"
    results_dir = "test_gan_results"
    for d in [models_dir, results_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    yield models_dir, results_dir
    for d in [models_dir, results_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

def test_save_and_load_checkpoint(temp_dirs):
    models_dir, _ = temp_dirs
    manager = GANManager(models_dir=models_dir)

    model_state = {"weights": [1, 2, 3]}
    optimizer_state = {"lr": 0.001}
    config = {"batch_size": 8}

    version = manager.save_checkpoint(
        model_state=model_state,
        optimizer_state=optimizer_state,
        epoch=1,
        config=config
    )

    # Verify files were created
    assert os.path.exists(os.path.join(models_dir, f"v{version}", "generator.pth"))
    assert os.path.exists(os.path.join(models_dir, f"v{version}", "training_state.pth"))
    assert os.path.exists(os.path.join(models_dir, f"v{version}", "config.json"))

    # Test loading
    checkpoint = manager.load_checkpoint(version)
    assert checkpoint["generator"] == model_state
    assert checkpoint["training_state"]["optimizer_state"] == optimizer_state
    assert checkpoint["config"] == config

def test_save_sample_images_and_metadata(temp_dirs):
    _, results_dir = temp_dirs
    manager = GANManager(results_dir=results_dir)
    images = torch.randint(0, 255, (2, 3, 8, 8), dtype=torch.uint8)
    manager.save_sample_images(images, epoch=1)
    sample_dir = os.path.join(results_dir, "samples", "epoch_0001")
    assert os.path.exists(os.path.join(sample_dir, "sample_0000.png"))
    assert os.path.exists(os.path.join(sample_dir, "meta_0000.json"))
    # Check metadata content
    with open(os.path.join(sample_dir, "meta_0000.json")) as f:
        meta = json.load(f)
    assert "epoch" in meta
    assert "sample_index" in meta

if __name__ == "__main__":
    pytest.main()
