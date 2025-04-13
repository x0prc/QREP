import torch
from syft.frameworks.torch import federated
from syft.frameworks.torch import hook
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class FederatedDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']

        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
        return inputs, torch.tensor(label)

class FederatedTrainer:
    def __init__(self,
               model: nn.Module,
               data_shards: list,
               num_rounds: int = 3):
        self.model = model
        self.data_shards = data_shards
        self.num_rounds = num_rounds
        self.hook = hook.torch.hook_torch()

    def train(self):
        workers = [federated.VirtualWorker(hook=self.hook, id=f"worker{i}")
                  for i in range(len(self.data_shards))]

        data_loaders = []
        for worker, data in zip(workers, self.data_shards):
            dataset = FederatedDataset(data, AutoTokenizer.from_pretrained("bert-base-cased"))
            data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
            data_loaders.append(data_loader.send(worker))

        for round in range(self.num_rounds):
            local_models = []
            for worker, data_loader in zip(workers, data_loaders):
                local_model = self.model.copy().send(worker)
                optimizer = optim.Adam(local_model.parameters(), lr=0.001)

                for batch in data_loader:
                    inputs, labels = batch
                    optimizer.zero_grad()
                    outputs = local_model(**inputs)
                    loss = nn.CrossEntropyLoss()(outputs.logits, labels)
                    loss.backward()
                    optimizer.step()

                local_models.append(local_model.copy().get())

            # Federated averaging
            with torch.no_grad():
                global_weights = [
                    sum(model.weight for model in local_models)/len(local_models)
                ]
            self.model.load_state_dict(global_weights)

        return self.model
