import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class DigitsDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class DigitsDataloader:
    def __init__(self, batch_size=32, test_size=0.3, random_state=42):
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        digits = load_digits()
        X = digits.data / 16.0
        y = digits.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)

        train_dataset = DigitsDataset(X_train.reshape(-1, 8, 8), y_train)
        test_dataset = DigitsDataset(X_test.reshape(-1, 8, 8), y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        if self.train_loader is None or self.test_loader is None:
            self.load_data()
        return self.train_loader, self.test_loader


# if __name__ == '__main__':
#     train_loader, test_loader = DigitsDataloader().get_loaders()