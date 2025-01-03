import torch
from sklearn.metrics import accuracy_score

from BreastData import BreastDataLoader
from DigitsData import DigitsDataloader
from MTrans import MTransModel


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        min_loss = float('inf')

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_data)

            # 计算损失
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            # 反向传播
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(batch_labels.numpy())
            y_pred.extend(predicted.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"测试集准确率: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    # digits_model = MTransModel(embed_size=8, num_heads=4, num_layers=2, num_classes=10, dropout=0.1)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(digits_model.parameters(), lr=0.001)
    # digits_loader = DigitsDataloader(batch_size=4)
    # digit_train_loader, digit_test_loader = digits_loader.get_loaders()
    # train_model(digits_model, digit_train_loader, criterion, optimizer)
    # evaluate_model(digits_model, digit_test_loader)

    breast_cancer_model = MTransModel(embed_size=30, num_heads=1, num_layers=2, num_classes=2, dropout=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(breast_cancer_model.parameters(), lr=0.001)
    breast_cancer_loader = BreastDataLoader(batch_size=4)
    breast_train_loader, breast_test_loader = breast_cancer_loader.get_loaders()
    train_model(breast_cancer_model, breast_train_loader, criterion, optimizer)
    evaluate_model(breast_cancer_model, breast_test_loader)

