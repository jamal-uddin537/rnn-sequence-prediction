"""Recurrent Neural Network (RNN/LSTM) for sequence prediction.

The model learns to predict the next value of a noisy sine wave from the
previous 30 values. Data is generated locally, so no download is needed.
"""
import math
import torch
import torch.nn as nn


def make_sine_dataset(n_samples=2000, seq_len=30, noise=0.05, seed=0):
    """Create (input window, next value) pairs from a noisy sine wave."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 40 * math.pi, n_samples + seq_len + 1)
    wave = torch.sin(t) + noise * torch.randn(t.shape, generator=g)
    x = torch.stack([wave[i:i + seq_len] for i in range(n_samples)]).unsqueeze(-1)
    y = wave[seq_len:seq_len + n_samples].unsqueeze(-1)
    return x, y


class SequenceRNN(nn.Module):
    """LSTM followed by a linear layer that outputs one value per sequence."""

    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        return self.head(out[:, -1, :])  # use the last time step


def train(epochs=30, batch_size=64, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    x, y = make_sine_dataset(seed=seed)
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = SequenceRNN()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(model(x_test), y_test).item()
            print(f"epoch {epoch:3d} | test MSE {test_loss:.5f}")

    return model


if __name__ == "__main__":
    trained = train()
    torch.save(trained.state_dict(), "rnn_model.pt")
    print("Saved model to rnn_model.pt")
