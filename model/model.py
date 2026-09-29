
import torch
import torch.nn as nn
import math
from typing import Optional
from einops import rearrange, einsum
from typing import Union
import utils



class Linear(nn.Module):
    '''Construct a linear transformation module w/o bias:
    Args:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
    '''
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # Create weight parameter W (shape: out_features × in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # Initialize weights using truncated normal
        std = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return = x @ self.weight.T #without einops
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    '''Embedding layer that maps token IDs to dense vectors:
    Args:
        num_embeddings: Size of the vocabulary. (vocab/num of tokens)
        embedding_dim: Dimension of each embedding vector. (d_model)
        device: Device on which to store the embedding matrix.
        dtype: Data type of the embedding matrix.

    Shape:
        Input: (...,)
        Weight: (num_embeddings, embedding_dim)
        Output: (..., embedding_dim)
    '''
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # Create embedding weight matrix and register as parameter
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        # Initialize with truncated normal
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''Lookup embedding vectors from tokens'''
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm). ## make this a kernal fusion for effeciency?

    RMSNorm normalizes inputs by their root mean square (second moment)
    rather than by mean and variance as in LayerNorm.

    Mathematical definition:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Key properties:
        - No mean subtraction (preserves directional information)
        - Only a learnable scale parameter (no bias)
        - Lower computational cost than LayerNorm
        - Widely used in modern LLMs (e.g., LLaMA, PaLM, Mistral)

    Args:
        d_model (int): Dimensionality of the input feature vector.
        eps (float): Small constant for numerical stability.
                              Defaults to 1e-5.
        dtype (optional): Data type of the learnable parameters.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Learnable scale parameter (gamma), shape: (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.

        Notes:
            - Normalization is applied over the last dimension only.
            - Computation is temporarily upcast to float32 for numerical
              stability, then cast back to the original dtype.
        """
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        # Mean square
        mean_square = (x_float * x_float).mean(dim=-1, keepdim=True)
        # Normalize
        x_norm = x_float * torch.rsqrt(mean_square + self.eps)
        return (x_norm * self.weight).to(orig_dtype)


class Positionwise_FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation.

    This module implements the SwiGLU variant of the Transformer
    feed-forward layer, used in modern large language models
    (e.g., LLaMA, PaLM, Mistral).

    The transformation is applied independently to each token
    (position-wise) and does not mix sequence positions.

    Mathematical form:
        FFN(x) = W2 ( SiLU(W1 x) ⊙ (W3 x) )

    Where:
        - ⊙ denotes element-wise multiplication
        - SiLU(x) = x * sigmoid(x)
        - W1, W3 expand from d_model → d_ff
        - W2 projects back from d_ff → d_model

    Args:
        d_model (int): Dimensionality of the input and output features.
        d_ff (int, optional): Hidden dimensionality of the FFN.
            If None, uses the common 8/3 * d_model rule and rounds
            to a multiple of 64 for hardware efficiency.
        dtype (optional): Data type of the parameters.
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            # Common SwiGLU sizing rule used in LLaMA-style models
            d_ff = int(8 * d_model / 3)
        d_ff = 64 * ((d_ff + 63) // 64)

        self.d_model = d_model

        # Sigmoid Linear Unit
        self.silu = SiLU()

        std = math.sqrt(2/(d_model+d_ff))
        # std = 1 / math.sqrt(3 * d_model)

        # Projection for the gated (SiLU) branch
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
        # Projection for the linear (gate) branch
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
        # Output projection back to model dimension
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU feed-forward network.
        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, sequence_length, d_model)
        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, sequence_length, d_model)
        Notes:
            - The same FFN is applied independently to each position.
            - No interaction occurs between tokens in this module.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected {self.d_model}, got {x.shape[-1]}")

        # SiLU activation
        silu_branch = self.silu(self.w1(x))
        # Linear gate branch
        gate = self.w3(x)
        # Element-wise gating and projection back to d_model
        return self.w2(silu_branch * gate)


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (SiLU) activation function.

    Also known as:
        - Swish
        - SiLU(x) = x * sigmoid(x)

    This activation is smooth, non-monotonic, and commonly used in
    modern Transformer FFNs (e.g., LLaMA, PaLM, Mistral) due to its
    favorable optimization properties compared to ReLU.

    Mathematical definition:
        SiLU(x) = x * σ(x)
        where σ(x) = 1 / (1 + exp(-x))

    Note:
        PyTorch already provides this as nn.SiLU, but this class
        demonstrates a manual implementation for clarity.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    This module applies the rotary positional embedding transformation described in:
        Su et al., 2021 — "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    For each token position i and each pair of embedding dimensions (2k, 2k+1),
    RoPE applies a 2D rotation:

        [x_2k']     [ cos(theta_i,k)  -sin(theta_i,k) ] [x_2k]
        [x_2k+1'] = [ sin(theta_i,k)   cos(theta_i,k) ] [x_2k+1]

    where:
        theta_i,k = i / (Theta^(2k/d_k))

    Instead of constructing the full block-diagonal rotation matrix, this
    implementation applies the rotation efficiently using elementwise operations.

    Notes
    -----
    - This module has no learnable parameters.
    - Cosine and sine values are precomputed up to `max_seq_len` and stored
      as non-persistent buffers.
    - Works with arbitrary batch dimensions.

    Args
    ----
    theta : float
        Base frequency constant (Θ in the paper).
    d_k : int
        Dimension of the query/key vectors (must be even).
    max_seq_len : int
        Maximum supported sequence length.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RotaryPositionalEmbedding.")

        self.d_k = d_k
        # self.max_seq_len = max_seq_len
        # self.theta = theta

        # Create frequency scaling factors for each dimension pair.
        # Shape: (d_k // 2,)
        #
        # inv_freq[k] = 1 / (theta^(2k/d_k))
        #
        # This corresponds to the denominator term in theta_i,k.
        k = torch.arange(0, d_k // 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (2 * k / d_k))

        # Token positions (0 ... max_seq_len-1)
        positions = torch.arange(max_seq_len, dtype=torch.float32)

        # Compute rotation angles:
        # angles[i, k] = i * inv_freq[k]
        # Shape: (max_seq_len, d_k // 2)
        # angles = torch.outer(positions, inv_freq)
        angles = einsum(positions, inv_freq, "i, j -> i j")

        # Precompute cosine and sine values
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Register as non-persistent buffers:
        # - Not trainable
        # - Not saved in state_dict
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., seq_len, d_k).
            The last dimension must equal d_k.

        token_positions : torch.Tensor
            Tensor of shape (..., seq_len) specifying the token index
            of each element along the sequence dimension.
            Must be broadcastable to the batch dimensions of x.

        Returns
        -------
        torch.Tensor
            Tensor of same shape as x with rotary positional
            embedding applied to the last dimension.
        """

        if x.shape[-1] != self.d_k:
            raise ValueError(f"Last dimension of x must be {self.d_k}, got {x.shape[-1]}")

        # Retrieve cos/sin values for the given token positions.
        # Resulting shape: (..., seq_len, d_k // 2)
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)

        # Split even and odd dimensions:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply rotation:
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # Interleave rotated pairs back into original dimension layout.
        # Stack last dim as (..., seq_len, d_k/2, 2) then flatten.
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)


class Multihead_self_attention(nn.Module):
    """
    Causal multi-head self-attention module with optional rotary positional embeddings (RoPE).

    This module implements the multi-head self-attention mechanism described in
    Vaswani et al. (2017), Section 3.2.2:

        MultiHeadSelfAttention(x) = W_O · MultiHead(W_Q x, W_K x, W_V x)

    where each attention head independently applies scaled dot-product attention
    with a causal (autoregressive) mask that prevents attending to future tokens.

    If a rotary positional embedding (RoPE) layer is provided, it is applied to
    the query and key tensors after splitting into heads and before computing
    attention scores. The same positional rotation is applied independently to
    each head.

    Args:
        d_model (int):
            Dimensionality of the input and output embeddings.
        num_heads (int):
            Number of attention heads. Must evenly divide `d_model`.
        positional_embedding_layer (Optional[nn.Module]):
            Optional module that applies rotary positional embeddings. This module
            must accept inputs of shape (..., seq_len, d_head) and corresponding
            token positions.
        dtype:
            Optional data type for parameter initialization.
    """

    def __init__(self, d_model: int, num_heads: int, positional_embedding_layer=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Projections (3 total, as in Vaswani et al.)
        self.q_proj = Linear(d_model, d_model, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, dtype=dtype)

        # Output projection
        self.output_proj = Linear(d_model, d_model, dtype=dtype)

        # Optional RoPE module
        self.positional_embedding_layer = positional_embedding_layer

    def forward(self,x: torch.Tensor):
        """
        Args:
            x:
                (batch_size, seq_len, d_model)
            token_positions:
                (seq_len,) or (batch_size, seq_len)
        Returns:
            output:
                (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads
        # (b, s, d_model) -> (b, h, s, d_head)
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE to NEW Q and K
        if self.positional_embedding_layer is not None:
            # if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

            Q = self.positional_embedding_layer(Q, token_positions)
            K = self.positional_embedding_layer(K, token_positions)

        # Flatten heads into batch dimension
        # (b, h, s, d) -> (b*h, s, d)
        # Q = rearrange(Q, "b h s d -> (b h) s d")
        # K = rearrange(K, "b h s d -> (b h) s d")
        # V = rearrange(V, "b h s d -> (b h) s d")

        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Scaled dot-product attention
        attn = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Restore heads
        # (b*h, s, d) -> (b, s, h*d)
        # attn = rearrange(attn, "(b h) s d -> b s (h d)", b=batch_size, h=self.num_heads)
        attn = rearrange(attn, "b h s d -> b s (h d)", b=batch_size, h=self.num_heads)

        # Output projection
        return self.output_proj(attn)

@staticmethod
def scaled_dot_product_attention(queries, keys, values, mask=None) -> torch.Tensor:
    """Computes Scaled Dot-Product Attention.

    Args:
        queries: Tensor of shape (batch, ..., seq_len_q, d_k).
        keys: Tensor of shape (batch, ..., seq_len_k, d_k).
        values: Tensor of shape (batch, ..., seq_len_k, d_v).
        mask: Boolean mask of shape (..., seq_len_q, seq_len_k).
            True indicates tokens to attend to; False indicates tokens to ignore.

    Returns:
        torch.Tensor: Weighted sum of values, shape (batch, ..., seq_len_q, d_v).
    """
    d_k = queries.size(-1)

    # Compute dot products between Q and K, scale by sqrt(d_k)
    # Equation: (batch, q, d) x (batch, k, d) -> (batch, q, k)
    scores = einsum(queries, keys, "... q d, ... k d -> ... q k") / math.sqrt(d_k)

    # Apply mask before softmax
    # if mask is not None:
    #     # Use a very large negative number so masked positions become 0 after softmax
    #     scores = scores.masked_fill(mask == 0, float("-inf"))
    # Mask out positions that should not be attended to
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Normalize scores across the key sequence length
    weights = utils.softmax(scores, dim=-1)

    # Apply weights to values
    # Equation: (batch, q, k) x (batch, k, v) -> (batch, q, v)
    return einsum(weights, values, "... q k, ... k v -> ... q v")


class TransformerBlock(nn.Module):
    """
    A transfomer block that contains two 'sublayers', one for the multihead self attention, and another for the feed-forward network.
    In each sublayer, first perform RMSNorm, then the main operation (MHA/FF), finally adding in the residual connection
    
    d_model: int, Dimensionality of the Transformer block inputs.
    num_heads: int, Number of heads to use in multi-head self-attention.
    d_ff: int, Dimensionality of the position-wise feed-forward inner layer.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, positional_embedding_layer: torch.Tensor | None):
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = Multihead_self_attention(d_model, num_heads, positional_embedding_layer)
        self.ffn = Positionwise_FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim (batch_size, seq_len, d_model)
        """
        ## out_1 dim: (batch_size, seq_len, d_model)
        attn_out = self.attn(self.ln1(x))
        out_1 = x + attn_out

        ## out_2 dim: (batch_size, seq_len, d_model)
        out_2 = out_1 + self.ffn(self.ln2(out_1))

        return out_2


class TransformerLM(nn.Module):
    """
    Full language model.
    """

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int,
                  num_heads: int, d_ff: int, rope_theta: float = 10000.0):
        """
        vocab_size: int, The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
        context_length: int, The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
        num_layers: int, The number of Transformer blocks to use.
        d_model: int, Dimensionality of the Transformer block inputs.
        num_heads: int, Number of heads to use in multi-head self-attention.
        d_ff: int, Dimensionality of the position-wise feed-forward inner layer.
        rope_theta: float, The RoPE Theta parameter.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length)

        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(f"{i}", TransformerBlock(d_model, num_heads, d_ff, positional_embedding_layer=rope))

        print(
            "vocab_size:", vocab_size,
            " context_length:", context_length,
            " num_layers:", num_layers,
            " d_model:", d_model,
            " num_heads", num_heads,
            " d_ff", d_ff
        )

    def forward(self, x: torch.Tensor):
        """
        x: Int[Tensor, "batch_size sequence_length"]
        output: Int[Tensor, "batch_size sequence_length vocab_size"], unnormalized prediction scores
        """

        ## embedding -> transformer blocks
        # out = self.layers(self.token_embeddings(x))
        out = self.token_embeddings(x)
        for layer in self.layers:
            out = layer(out)
            
        ## norm, linear
        return self.lm_head(self.ln_final(out))
