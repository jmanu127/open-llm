import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply the softmax operation to a tensor along a specified dimension.

    Softmax converts a vector of arbitrary real-valued scores into a probability
    distribution by exponentiating each value and normalizing by the sum of
    exponentials. The output values are all non-negative and sum to 1 along the
    specified dimension.

    For numerical stability, this implementation subtracts the maximum value
    along the softmax dimension before applying the exponential. This does not
    change the result of softmax, but prevents overflow when input values are
    large.

    Args:
        x (torch.Tensor): Input tensor containing unnormalized scores.
        dim (int): Dimension along which to apply the softmax operation.

    Returns:
        torch.Tensor: A tensor of the same shape as `x`, where the values along
        dimension `dim` form a valid probability distribution (sum to 1).
    """
    # Subtract the maximum value along `dim` for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values

    # Exponentiate the shifted values
    x_exp = torch.exp(x - x_max)

    # Normalize by the sum of exponentials along `dim`
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean cross-entropy loss from unnormalized logits.

    This function implements numerically stable cross-entropy using the
    log-sum-exp trick. It is equivalent to:

        torch.nn.functional.cross_entropy(logits, targets)

    but written out explicitly.

    Args:
        logits (torch.Tensor):
            Unnormalized model outputs of shape (..., vocab_size).
            The last dimension represents class scores (logits).
            All preceding dimensions are treated as batch dimensions.

        targets (torch.Tensor):
            Integer class indices of shape (...), matching the batch
            shape of `logits` (i.e., same shape as `logits` without
            the final vocab dimension). Each value must be in
            [0, vocab_size - 1].

    Returns:
        torch.Tensor:
            A scalar tensor containing the mean cross-entropy loss
            over all batch elements.

    Notes:
        - Numerical stability is ensured by subtracting the maximum
          logit value along the class dimension before exponentiation
          (log-sum-exp trick).
        - The loss for each element is:

              -log( exp(logit_target) / sum_j exp(logit_j) )

          which is equivalent to:

              -log_softmax(logits)[target]

        - To compute perplexity from the returned loss:
              
              perplexity = torch.exp(loss)
    """

    # 1. subtract max logit for numerical stability
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits

    # 2. compute log-sum-exp
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))

    # 3. gather correct class logit
    target_logits = shifted.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    # 4. compute negative log likelihood
    loss = -target_logits + log_sum_exp

    # 5. average over all batch dimensions
    # perplexity = torch.exp(loss)
    return loss.mean()


def gradient_clipping(parameters, max_norm, eps=1e-6):
    """
    Clip gradients in-place so that the global L2 norm
    does not exceed max_norm.

    Args:
        parameters: iterable of nn.Parameter
        max_norm: float, maximum allowed L2 norm
        eps: small constant for numerical stability (default 1e-6)
    """

    # Collect gradients
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return

    # Compute global L2 norm
    total_norm_sq = 0.0
    for g in grads:
        total_norm_sq += torch.sum(g.detach() ** 2)

    total_norm = torch.sqrt(total_norm_sq)

    # If clipping is needed
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)