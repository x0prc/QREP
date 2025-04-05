import math
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from stylegan2_pytorch import Trainer
import syft as sy
from typing import List, Dict

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
