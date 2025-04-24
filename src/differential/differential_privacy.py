import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ContextAwareDP:
    def __init__(self, epsilon_base=0.1):
        self.epsilon_base = epsilon_base
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=10)

    def calculate_privacy_budget(self, context_score, diversity_metric):
        return self.calculate_epsilon(context_score, diversity_metric)

    def add_laplace_noise(self, data, sensitivity, epsilon):
        return self.add_noise(data, sensitivity=sensitivity, epsilon=epsilon)

    def apply_differential_privacy(self, financial_data, epsilon=0.1):
        return self.add_noise(financial_data, epsilon=epsilon)

    def process_data(self, text_data, diversity_metric):
        context_score = self.analyze_text(text_data)
        adjusted_epsilon = self.calculate_epsilon(context_score, diversity_metric)
        return {"adjusted_epsilon": adjusted_epsilon}

    def calculate_epsilon(self, context_score, diversity):
        """Calculate adaptive privacy budget"""
        return (self.epsilon_base * (1 + context_score/10)**-1) + (diversity/5)

    def analyze_text(self, text):
        """Classify text sensitivity using BERT"""
        inputs = self.tokenizer(text, return_tensors="pt")
        return torch.argmax(self.model(**inputs).logits).item()

    def add_noise(self, data, sensitivity=1, epsilon=0.1):
        """Apply Laplace mechanism to numeric data"""
        return data + np.random.laplace(scale=sensitivity/epsilon, size=data.shape)

if __name__ == "__main__":
    dp = ContextAwareDP()

    text = "Large financial transaction: $1,000,000"
    context_score = dp.analyze_text(text)

    financial_data = np.array([100.0, 200.0, 50.0])
    noisy_data = dp.add_noise(financial_data, epsilon=0.5)

    print(f"Adjusted epsilon: {dp.calculate_epsilon(context_score, 4.2)}")
    print(f"Noisy data: {noisy_data}")
