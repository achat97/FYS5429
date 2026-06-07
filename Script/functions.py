import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



def plot_spectrogram(t, f, zxx):

    """
    Plots a spectrogram with time on the x-axis and frequency on the y-axis.
    The magnitude is assumed to be in dB

    ----------

    Parameters:
        t (ndarray) - 1-D array of time bin values (in seconds).
        f (ndarray) - 1-D array of frequency bin values (in Hz).
        zxx (ndarray) - 2-D array containing the spectrogram magnitudes (in dB).

    Returns:
        None
    """

    plt.pcolormesh(t, f, zxx, shading='gouraud', cmap='plasma')
    plt.colorbar(label='dB')
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.show()


def array_to_tensor(X_input):

    """
    Converts a list or array of spectrogram samples into a stacked PyTorch tensor suitable for batched model input.
    Each spectrogram is also given a leading channel dimension before all samples are stacked along the batch dimension.

    ----------

    Parameters:
        X_input (ndarray or list) - sequence of samples, where each sample has zxx as its third element.

    Returns:
        X (Tensor) - 4-D float tensor of shape (batch_size, 1, frequency_bins, time_bins).
    """

    X = []

    for sample in X_input:
        X.append(torch.from_numpy(sample[2]).float().unsqueeze(0))

    X = torch.stack(X)

    return X


class ReshapeForLSTM(nn.Module):

    """
    Reshapes a 4-D CNN feature map into a 3-D sequence suitable for an LSTM.
    The input tensor has shape (batch_size, channels, height, width) and is reshaped to
    (batch_size, width, channels * height) by treating the width axis as the time-step axis and
    flattening channels and height into a single feature dimension.

    ----------

    Forward parameters:
        x (Tensor) - 4-D input tensor of shape (batch_size, channels, height, width).

    Forward returns:
        (Tensor) - 3-D output tensor of shape (batch_size, width, channels * height).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        return x.permute(0, 3, 1, 2).reshape(batch_size, width, -1)


def save_padded_sequences(y, filename='padded_sequences.npy', target_length=10):

    """
    Pads a collection of variable-length target sequences to a fixed length and saves the resulting
    stacked array as a .npy file. All padded sequences are then stacked into a single 3-D array.

    ----------

    Parameters:
        y (sequence) - input sequence of variable-length 2-D arrays to pad.
        filename (str) - default set to 'padded_sequences.npy'. Name of the output .npy file.
        target_length (int) - default set to 10. Target sequence length for all sequences.

    Returns:
        padded_numpy (ndarray) - 3-D array of shape (batch_size, target_length, input_size).
    """

    sequences = []
    for i in range(len(y)):
        seq = torch.tensor(y[i], dtype=torch.float32)

        padding = torch.zeros(target_length - len(seq), seq.shape[1], dtype=torch.float32)
        seq = torch.cat([seq, padding], dim=0)

        sequences.append(seq)

    padded_sequences = torch.stack(sequences, dim=0)

    padded_numpy = padded_sequences.numpy()
    np.save(filename, padded_numpy)

    #print(f"Saved padded sequences to '{filename}'")
    #print(f"Shape: {padded_numpy.shape}")  
    return padded_numpy
