import math
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from stylegan2_pytorch import Trainer
import syft as sy
from typing import List, Dict
from this import d

class ContextAwareDP:
    def __init__(self, epsilon_base: float = 0.1):
        self.epsilon_base = epsilon_base
        self._init_context_analyzer()
        self.init_gan_trainer()
        self.federated_models = {}

    def _init_context_analyzer(self):
            """Load BERT-based sensitivity classifier with PyTorch backend"""
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-cased",
                num_labels=10
            )
            self.context_pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                framework="pt"
            )

    def _init_gan_trainer(self):
        """Initialize GAN trainer"""
        self.gan_trainer = Trainer(
            name = "health_records_synth",
            data = "./healthcare_dataset",
            results_dir = "./gan_results",
            models_dir = "./gan_models",
            batch_size = 8,
            gradient_accumulate_every = 4,
            image_size = 256,
            network_capacity = 18
        )

    def calculate_privacy_budget(self, context_score: int, diversity_metric: float) -> float:
        return (self.epsilon_base * math.pow(1 + context_score/10, -1) + diversity_metric/5)
