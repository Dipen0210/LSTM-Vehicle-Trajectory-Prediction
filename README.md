# 🚗 Vehicle Trajectory Prediction Using Manually Implemented LSTM

This repository contains a custom implementation of Long Short-Term Memory (LSTM) networks from scratch in PyTorch to forecast future vehicle positions based on historical trajectory data. Unlike standard LSTM layers, this project builds the LSTM cell using only matrix operations and activation functions, offering a transparent and educational perspective on sequence modeling.

---

## 📌 Project Overview

- **Objective**: Predict 2D coordinates (Local_X, Local_Y) of a vehicle for the next 5 time steps, using a 62-step time series input.
- **Approach**: Implement an LSTM manually, without using `torch.nn.LSTM`, and optimize performance through hyperparameter tuning.
- **Dataset**: Real-world vehicle trajectory dataset containing over 9,000 files. Each file includes 67 time steps of 12 features.

---

## 🧠 Model Architecture

- **Input Features**: 12 per timestep
- **Prediction Horizon**: 5 future steps (each with 2D position)
- **Custom LSTM Cell**: Built from scratch using PyTorch tensors, `sigmoid`, `tanh`, and linear projections.
- **Autoregressive Forecasting**: Outputs from one step are fed as inputs to the next (recursive prediction).

---

## 🧪 Experiments

We trained six different models by varying:

| Model | Hidden Size | Dropout | Epochs | Test RMSE |
|-------|-------------|---------|--------|-----------|
| model_1 | 128       | 0.2     | 200    | 0.2498    |
| model_2 | 256       | 0.2     | 100    | 0.2598    |
| model_3 | 512       | 0.2     | 100    | 0.2689    |
| model_4 | 512       | 0.3     | 80     | 0.2462    |
| model_5 | 1024      | 0.2     | 100    | 0.2707    |
| **model_6** | **1024** | **0.4** | **60** | **0.2375** |

> ✅ **Model_6** performed best with a test RMSE of 0.2375.
