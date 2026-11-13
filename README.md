# RNN Sequence Prediction

An LSTM (a type of Recurrent Neural Network) that learns to predict the next value of a noisy sine wave from the previous 30 values.

## What it shows

- How a recurrent model processes a sequence one step at a time
- Building an `nn.LSTM` with a linear output head in PyTorch
- Training with MSE loss and the Adam optimizer

## Setup

```bash
pip install torch
```

## Run

```bash
python rnn.py
```

The script trains for 30 epochs, prints the test error, and saves the model to `rnn_model.pt`. No dataset download is needed; the data is generated in code.

## Files

- `rnn.py`: data generation, the LSTM model, and the training loop
- `requirements.txt`: Python dependencies
