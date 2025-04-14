import pandas as pd
import numpy as np
from typing import Tuple, List

class FinancialTransactionsDataset:
    def __init__(self, file_path="./data/financial_transactions/transactions.csv"):
        self.file_path = file_path
        self.data = None
        self.features = None
        self.labels = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    def preprocess(self):
        if self.data is None:
            self.load_data()
        self.features = self.data.drop(['time', 'class'], axis=1).values
        self.labels = self.data['class'].values
        self.features[:, 0] = (self.features[:, 0] - np.mean(self.features[:, 0])) / np.std(self.features[:, 0])
        return self.features, self.labels

    def add_laplace_noise(self, epsilon=0.1):
        if self.features is None:
            self.preprocess()
        noise = np.random.laplace(0, 1/epsilon, self.features.shape[0])
        self.features[:, 0] += noise
        return self.features

    def split_into_shards(self, num_shards=4):
        if self.data is None:
            self.load_data()
        return [self.data.iloc[i::num_shards] for i in range(num_shards)]

    def generate_synthetic_data(self, num_samples=1000):
        if self.data is None:
            self.load_data()
        synthetic_data = {
            'time': np.random.uniform(0, 172000, num_samples),
            'amount': np.random.normal(self.data['amount'].mean(), self.data['amount'].std(), num_samples),
            'class': np.random.randint(0, 2, num_samples)
        }
        for i in range(1, 29):
            col_name = f'v{i}'
            synthetic_data[col_name] = np.random.normal(self.data[col_name].mean(),
                                                       self.data[col_name].std(),
                                                       num_samples)
        return pd.DataFrame(synthetic_data)

if __name__ == "__main__":
    dataset = FinancialTransactionsDataset()
    features, labels = dataset.preprocess()
    noisy_features = dataset.add_laplace_noise()
    shards = dataset.split_into_shards()
    synthetic_df = dataset.generate_synthetic_data()
