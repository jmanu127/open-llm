import torch

def save_checkpoint(model, optimizer, iteration, out):
    """
    Save training checkpoint.

    Args:
        model (torch.nn.Module):
            Model whose parameters should be saved.

        optimizer (torch.optim.Optimizer):
            Optimizer containing training state (e.g. Adam moments).

        iteration (int):
            Current training iteration/step number.

        out (str | os.PathLike | BinaryIO):
            Destination path or file-like object.
    """

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer=None):
    """
    Load a training checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO):
            Path or file-like object containing the checkpoint.

        model (torch.nn.Module):
            Model to restore parameters into.

        optimizer (torch.optim.Optimizer):
            Optimizer to restore state into.

    Returns:
        int:
            The iteration number stored in the checkpoint.
    """

    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state"])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint["iteration"]