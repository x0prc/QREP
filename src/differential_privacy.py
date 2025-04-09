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


    def federated_pattern_analysis(self, data_shards: List[Dict], num_rounds: int = 3):
        hook = sy.TorchHook(torch)
        workers = [sy.VirtualWorker(hook, id=f"worker{i}")
            for i in range(len(data_shards))]

        global_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")

        for round in range(num_rounds):
            local_models = []
            for worker, data in zip(workers, data_shards):
                local_model = global_model.copy().send(worker)
                local_models.append(local_model.copy().get())

            with torch.no_grad():
                global_weights = [
                    sum(model.weight for model in local_models)/len(local_models)
                ]
                global_model.load_state_dict(global_weights)

            self.federated_model = global_model

    def generate_synthetic_records(self, num_samples: int = 100):
        if not self.gan_trainer.is_trained:
            self.gan_trainer.train()

        return self.gan_trainer.generate(num_samples)

    def process_data(self, text_data: str, diversity_metric: float) -> dict:
        context_result = self.context_pipeline(text_data)
        context_score = int(context_result[0]['label'].split('_')[-1])

        adjusted_epsilon = self.calculate_privacy_budget(
            context_score, diversity_metric)
        if context_score >= 7:
                    synthetic_data = self.generate_synthetic_records()
                    return {
                        "original_data": text_data,
                        "context_score": context_score,
                        "adjusted_epsilon": adjusted_epsilon,
                        "synthetic_data": synthetic_data[:3],
                        "risk_level": "high"
                    }
        else:
                    return {
                        "original_data": text_data,
                        "context_score": context_score,
                        "adjusted_epsilon": adjusted_epsilon,
                        "risk_level": "low"
                    }
