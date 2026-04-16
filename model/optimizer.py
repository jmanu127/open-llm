import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Implements the AdamW optimization algorithm.

    AdamW is a variant of Adam that decouples weight decay from the
    gradient-based update. This improves regularization compared to
    traditional L2 regularization applied directly to gradients.

    The update rule is:

        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2

        m̂_t = m_t / (1 - β1^t)
        v̂_t = v_t / (1 - β2^t)

        θ ← θ - lr * m̂_t / (sqrt(v̂_t) + eps)

        θ ← θ - lr * weight_decay * θ     (decoupled weight decay)

    References:
        - "Decoupled Weight Decay Regularization"
          (Loshchilov & Hutter, 2019)

    Args:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.

        lr (float, optional):
            Learning rate (default: 1e-3).

        betas (Tuple[float, float], optional):
            Coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).

        eps (float, optional):
            Term added to the denominator for numerical stability
            (default: 1e-8).

        weight_decay (float, optional):
            Decoupled weight decay coefficient (default: 0.0).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        """
        Initialize the AdamW optimizer.

        Raises:
            ValueError: If any hyperparameter is outside its valid range.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
        if eps < 0.0:
            raise ValueError("Invalid epsilon value")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        This method:
            1. Updates exponential moving averages of gradients (m)
               and squared gradients (v).
            2. Applies bias correction to both moments.
            3. Updates parameters using the Adam update rule.
            4. Applies decoupled weight decay.

        Args:
            closure (callable, optional):
                A closure that reevaluates the model and returns the loss.
                Required for some second-order optimizers but optional here.
                If provided, it will be executed with gradients enabled.

        Returns:
            loss (torch.Tensor or None):
                The loss returned by the closure if provided,
                otherwise None.

        Notes:
            - Gradients must already be computed via `loss.backward()`
              before calling `step()`.
            - This method runs under `torch.no_grad()` because parameter
              updates should not be tracked by autograd.
            - State is stored per parameter:
                * step: current timestep
                * m: first moment estimate
                * v: second moment estimate
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]

                state["step"] += 1
                t = state["step"]

                # ---- Update biased first and second moments ----
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # ---- Bias correction ----
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # ---- Parameter update ----
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)

                # ---- Decoupled weight decay ----
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss
    

def learning_rate_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    """
    Compute cosine learning rate with warmup.

    Args:
        t (int): current step (starts at 0 or 1 — both handled correctly)
        alpha_max (float): maximum learning rate
        alpha_min (float): minimum learning rate
        T_w (int): number of warmup steps
        T_c (int): number of cosine annealing steps (end of cosine phase)

    Returns:
        float: learning rate at step t
    """

    # ---- Warmup phase ----
    if T_w > 0 and t < T_w:
        return alpha_max * (t / T_w)

    # ---- Cosine annealing phase ----
    if t <= T_c:
        # If no warmup, treat Tw = 0 safely
        denom = max(1, T_c - T_w)
        progress = (t - T_w) / denom
        cosine_term = math.cos(progress * math.pi)
        return alpha_min + 0.5 * (1 + cosine_term) * (alpha_max - alpha_min)

    # ---- Post-annealing ----
    return alpha_min


