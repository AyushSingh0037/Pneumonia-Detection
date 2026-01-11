import torch

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stop_training = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), "weights/best_model.pth")
        else:
            self.counter += 1
            print(f"Validation not improving ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.stop_training = True
