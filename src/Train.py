import torch
from earlystopping import EarlyStopping
from Validation import validate

def train_model(model, train_loader, val_loader, optimizer, criterion, device):
    early_stopping = EarlyStopping(patience=3)

    for epoch in range(20):
        model.train()
        train_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.stop_training:
            print("Early stopping triggered")
            break
