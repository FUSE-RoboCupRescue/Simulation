import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train(model: nn.Module,
          device: str,
          train_loader: DataLoader,
          val_loader: DataLoader,
          n_epochs: int = 10,
          checkpoint_every: int = 1000,
          lr: float = 1e-3,
          save_logs: bool=False):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    train_losses = []
    val_losses = []
    loss_fn = nn.MSELoss()

    train_info = {
        "val_size": len(val_loader) * val_loader.batch_size,
        "train_size": len(train_loader) * train_loader.batch_size,
        "val_batch_size": val_loader.batch_size,
        "train_batch_size": train_loader.batch_size,
        "n_epochs": n_epochs,
        "learning_rate": lr,
        "optimizer": "AdamW",
        "loss_fn": "MSELoss",
    }

    for epoch in range(n_epochs):

        # Train step
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(X)
            loss = loss_fn(out, y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()  
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += loss_fn(out, y).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print losses for this epoch
        print(f"Epoch {epoch + 1},\t Train Loss: {np.mean(train_losses):.6f},\t Val Loss: {val_loss:.6f}")
        train_losses = []

        if save_logs and (epoch+1) % checkpoint_every == 0:
            train_info[f"epoch_{epoch+1}_loss_val: "] = val_loss
            train_info[f"epoch_{epoch+1}_loss_train: "] = train_loss
            with open(f'{model.path}/train_info.json', 'w', encoding='utf-8') as f:
                json.dump(train_info, f)
            
            model.save(epoch+1)

    if save_logs:
        model.save(n_epochs)
        with open(f'{model.path}/train_info.json', 'w', encoding='utf-8') as f:
            json.dump(train_info, f)
            
    return model
       