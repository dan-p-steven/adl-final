import torch

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, save_path="./models/best_model.pth"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta 
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:

            self.best_loss = val_loss
            self.save_model(model) 

        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_model(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"EARLY STOPPING TRIGGERED! {self.best_loss}")

    def save_model(self, model):
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f"\tBest Score! Model saved to {self.save_path}")
