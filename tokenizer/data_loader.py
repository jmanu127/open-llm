import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    Sample a batch of training sequences for next-token prediction.

    Args:
        x (np.ndarray):
            1D array of token IDs of shape (n,). Can be a normal numpy
            array or a memory-mapped array (np.memmap).

        batch_size (int):
            Number of sequences in the batch (B).

        context_length (int):
            Length of each input sequence (m).

        device (str):
            PyTorch device to place tensors on
            (e.g. 'cpu', 'cuda:0', 'mps').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            inputs:  (batch_size, context_length)
            targets: (batch_size, context_length)

            inputs[b]  = [x_i, ..., x_{i+m-1}]
            targets[b] = [x_{i+1}, ..., x_{i+m}]
    """

    n = len(x)

    # Random start indices
    ix = np.random.randint(0, n - context_length, size=batch_size)
    # ix = np.random.choice(n-context_length, batch_size, replace=False)
    # Build batch
    inputs = np.stack([x[i : i + context_length] for i in ix])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in ix])

    # Convert to tensors and move to device
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets

