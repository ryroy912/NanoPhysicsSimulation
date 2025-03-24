import torch
import numpy as np
from models.transformer import TransformerModel
import matplotlib.pyplot as plt

def test_model(model, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    predicted_trajectory = [test_data[0].unsqueeze(0)]
    x_input = test_data[0].unsqueeze(0).to(device)

    for _ in range(len(test_data) - 1):
        x_input = model(x_input.unsqueeze(1)).squeeze(1).detach()
        predicted_trajectory.append(x_input.cpu())

    return torch.cat(predicted_trajectory, dim=0).numpy()
