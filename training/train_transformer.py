import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import TransformerModel

def train_model(X_train, Y_train, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(50):
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            Y_pred = model(X_batch.unsqueeze(1)).squeeze(1)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.5f}")

    return model
