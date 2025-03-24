# Lennard-Jones Simulation with Transformer-Based Prediction

This project simulates the interaction of two atoms using the Lennard-Jones potential and trains deep learning models (like Transformers) to predict atomic trajectories over time. The project is modular and supports multiple simulators, models, and visualization techniques.


## Project Structure
LennardJones-Simulation/ ├── main.py # Central entry script (CLI enabled) ├── requirements.txt # Python dependencies ├── README.md # This file

├── simulators/ │ └── simulator_lj.py # Lennard-Jones simulator

├── models/ │ ├── transformer.py # Transformer model definition │ ├── lstm.py # (Optional) LSTM model │ └── init.py

├── training/ │ └── train_transformer.py # Training loop for Transformer

├── testing/ │ ├── test_transformer.py # Autoregressive prediction │ └── visualization.py # Visualization utilities

├── utils/ │ ├── data_utils.py # Min-Max normalization and data wrangling │ └── config.py # Simulation config

## Install Dependencies
Make sure you have Python ≥ 3.7 installed.

pip install -r requirements.txt

## Run the Project via Command Line
Train and test the Transformer model using the Lennard-Jones simulator:

python main.py --simulator lj --model transformer --train --test
Available options:

--simulator lj : Lennard-Jones simulator
--model transformer : Transformer-based model
--train : Train the model
--test : Run autoregressive prediction and visualization
