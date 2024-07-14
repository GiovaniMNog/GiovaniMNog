# MNIST Multi-Layer Perceptron (MLP) with PyTorch

This project demonstrates the implementation of a Multi-Layer Perceptron (MLP) using PyTorch for handwritten digit classification on the MNIST dataset. The code supports training on both CPU and GPU (if available) and saves the trained model along with its parameters.

## Project Description

The goal of this project is to build an MLP for MNIST digit classification, explore the benefits of using CUDA for training acceleration, and save the trained model for future use. The model is trained and evaluated in terms of loss and accuracy, and the results are visualized through loss and accuracy graphs over the training epochs.

## Code Structure

- `MLP`: Definition of the MLP model class with two hidden layers.
- `train()`: Function for training the model.
- `val()`: Function for validating the model.
- Train the model for 20 epochs with loss and accuracy calculation.
- Save the trained model and its parameters to a file.
- Visualize the results through loss and accuracy plots.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib

You can install the dependencies using pip:

```bash
pip install torch torchvision matplotlib
