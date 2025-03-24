import os
import torch
import numpy as np
from utils.config import SIMULATION_CONFIG
from simulators.simulator_lj import velocity_verlet, Particle, Vector3D
from models.transformer import TransformerModel
from training.train_transformer import train_model
from testing.test_transformer import test_model
from utils.data_utils import prepare_data, min_max_scale, inverse_min_max_scale

def main():
    # Step 1: Generate Simulation Data
    print("Generating simulation data...")
    sigma, epsilon, dt, total_time = SIMULATION_CONFIG["sigma"], SIMULATION_CONFIG["epsilon"], SIMULATION_CONFIG["dt"], SIMULATION_CONFIG["total_time"]
    
    initial_positions = [(Vector3D(x1, y1, z1), Vector3D(x2, y2, z2)) 
                         for x1 in [0, 1] for y1 in [0, 1] for z1 in [0, 1] 
                         for x2 in [0, 1] for y2 in [0, 1] for z2 in [0, 1] 
                         if (x1, y1, z1) != (x2, y2, z2)]
    train_positions = initial_positions[:19]

    dataset_train = [velocity_verlet(
        [Particle(p1, Vector3D(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))),
         Particle(p2, Vector3D(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)))],
        epsilon, sigma, dt, total_time) for p1, p2 in train_positions]

    # Step 2: Prepare Data
    print("Preparing data...")
    X_train, Y_train, dataloader, scaler = prepare_data(dataset_train, apply_scaling=True)

    # Step 3: Train Model
    print("Training model...")
    model = train_model(X_train, Y_train, dataloader)

    # Step 4: Save Model
    model_path = "models/transformer_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Step 5: Test Model
    print("Testing model...")
    predicted_trajectory = test_model(model, dataset_train[0])

    # Step 6: Scale Back Predictions to Original Range
    predicted_trajectory = inverse_min_max_scale(predicted_trajectory, scaler)

    print("Visualization complete.")

if __name__ == "__main__":
    main()
