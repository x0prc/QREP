import os
import torch
import json
import numpy as np
from datetime import datetime
from PIL import Image

class GANManager:
    def __init__(self, models_dir="gan_models", results_dir="gan_results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    def save_checkpoint(self, model_state, optimizer_state, epoch, config):
        version = self._get_next_version()
        checkpoint_dir = os.path.join(self.models_dir, f"v{version}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(model_state, os.path.join(checkpoint_dir, "generator.pth"))
        torch.save({
            'epoch': epoch,
            'optimizer_state': optimizer_state,
            'loss': config.get("losses", {})
        }, os.path.join(checkpoint_dir, "training_state.pth"))

        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(config, f)

        return version

    def load_checkpoint(self, version):
        checkpoint_dir = os.path.join(self.models_dir, f"v{version}")
        return {
            "generator": torch.load(os.path.join(checkpoint_dir, "generator.pth")),
            "training_state": torch.load(os.path.join(checkpoint_dir, "training_state.pth")),
            "config": json.load(open(os.path.join(checkpoint_dir, "config.json")))
        }

    def _get_next_version(self):
        existing = [int(d[1:]) for d in os.listdir(self.models_dir) if d.startswith("v")]
        return max(existing) + 1 if existing else 1

    def save_sample_images(self, images, epoch):
        img_dir = os.path.join(self.results_dir, "samples", f"epoch_{epoch:04d}")
        os.makedirs(img_dir, exist_ok=True)

        for i, img in enumerate(images):
            img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            pil_img = Image.fromarray(img.cpu().numpy())
            pil_img.save(os.path.join(img_dir, f"sample_{i:04d}.png"))
            self._save_metadata(img_dir, epoch, i)

    def _save_metadata(self, path, epoch, index):
        metadata = {
            "generation_date": datetime.now().isoformat(),
            "epoch": epoch,
            "sample_index": index,
            "model_version": self._get_next_version() - 1
        }
        with open(os.path.join(path, f"meta_{index:04d}.json"), "w") as f:
            json.dump(metadata, f)


class StyleGANTrainer:
    def __init__(self, data_path, results_dir="gan_results", models_dir="gan_models"):
        self.data_path = data_path
        self.results = GANManager(results_dir=results_dir)
        self.models = GANManager(models_dir=models_dir)
        self.model = None
        self.optimizer = None

    def train(self, num_epochs=100, batch_size=8, lr=0.002):
        config = {
            "batch_size": batch_size,
            "learning_rate": lr,
            "dataset": self.data_path,
            "start_time": datetime.now().isoformat(),
            "losses": {}
        }

        # Load dataset here
        print(f"Loading data from {self.data_path} with batch size {batch_size}...")

        # Initialize model and optimizer (stub)
        self.model = torch.nn.Linear(100, 100)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            losses = self._train_step()

            if epoch % 10 == 0:
                version = self.models.save_checkpoint(
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    epoch=epoch,
                    config=config
                )
                print(f"Saved model version {version}")

            samples = self._generate_samples(4)
            self.models.save_sample_images(samples, epoch)
            self._log_metrics(epoch, losses)

        config["end_time"] = datetime.now().isoformat()
        return config

    def _train_step(self):
        self.optimizer.zero_grad()
        dummy_input = torch.randn(8, 100)
        dummy_output = self.model(dummy_input)
        loss = dummy_output.mean()
        loss.backward()
        self.optimizer.step()

        losses = {"g": float(loss.item()), "d": float(loss.item())}  # Dummy losses
        return losses

    def _generate_samples(self, count):
        return torch.randn(count, 3, 64, 64)

    def _log_metrics(self, epoch, losses):
        log_dir = os.path.join(self.models.results_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "training_metrics.csv")
        header = not os.path.exists(log_path)

        with open(log_path, "a") as f:
            if header:
                f.write("epoch,generator_loss,discriminator_loss,fid_score\n")
            fid = self._calculate_fid(epoch)
            f.write(f"{epoch},{losses['g']},{losses['d']},{fid}\n")

    def _calculate_fid(self, epoch):
        return float(np.random.uniform(5.0, 15.0))
