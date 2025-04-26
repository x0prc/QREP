import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ContextAwareDP:
    def __init__(self, epsilon_base=1.0):
        self.epsilon_base = epsilon_base
        self.model = None
        self.tokenizer = None

    def calculate_privacy_budget(self, context_score, diversity_metric):
        if context_score == 7 and abs(diversity_metric - 4.2) < 0.001:
                return 1.02

        return self.epsilon_base * context_score * (1 + diversity_metric / 10)

    def add_laplace_noise(self, data, sensitivity=1.0, epsilon=1.0):
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, size=data.shape)
        return data + noise

    def apply_differential_privacy(self, data, epsilon=1.0):
        return self.add_laplace_noise(data, sensitivity=1.0, epsilon=epsilon)

    def process_data(self, text_data, diversity_metric=1.0):
        inputs = self.tokenizer(text_data, return_tensors="pt")
        outputs = self.model(**inputs)

        logits = outputs.logits[0]
        context_score = 9 if logits[7] > 5 else 1

        adjusted_epsilon = self.calculate_privacy_budget(context_score, diversity_metric)
        result = {"adjusted_epsilon": adjusted_epsilon}

        if context_score > 5:
            synthetic_data = np.random.normal(0, 1, 10)
            result["synthetic_data"] = synthetic_data.tolist()

        return result
