import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer


class BreastDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class BreastDataLoader:
    def __init__(self, batch_size=32, test_size=0.2, random_state=42):
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        breast_cancer = load_breast_cancer()
        x = breast_cancer.data
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        y = breast_cancer.target

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)

        train_dataset = BreastDataset(x_train, y_train)
        test_dataset = BreastDataset(x_test, y_test)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

    def get_loaders(self):
        if self.train_loader is None or self.test_loader is None:
            self.load_data()
        return self.train_loader, self.test_loader