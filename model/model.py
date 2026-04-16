
import torch
import torch.nn as nn
import math
from typing import Optional
from einops import rearrange, einsum
from typing import Union
from cs336_basics import utils

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

# model = NeuralNetwork().to(device)
# print(model)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        Construct a linear transformation module. This function should accept the 
        following parameters:

        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        # Create weight parameter W (shape: out_features × in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Initialize weights using truncated normal
        std = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
        # nn.init.trunc_normal_(self.weight)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Apply the linear transformation to the
        input.
        '''
        # x: (..., in_features)
        # W: (out_features, in_features)
        # Output: (..., out_features)
        # output = x @ self.weight.T
        # output = einsum(x, self.weight, "... d_in, d_in d_out -> ... d_out")
        # output = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):
    """
    embedding layer that maps integer token IDs into a vector space of dimension
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: int, Size of the vocabulary
        embedding_dim: int, Dimension of the embedding vectors i.e. d_model
        device: torch.device | None = None, Device to store the parameters on 
        dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()

        # Create embedding weight matrix and register as parameter
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # Initialize with truncated normal
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
   

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (...,) LongTensor of token indices
        returns: (..., embedding_dim)
        """
        # return self.weight[token_ids]
        return self.weight.index_select(0, token_ids.reshape(-1)).view(*token_ids.shape, -1)
    

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

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
        eps (float, optional): Small constant for numerical stability.
                              Defaults to 1e-5.
        device (optional): Device on which parameters are allocated.
        dtype (optional): Data type of the learnable parameters.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable scale parameter (gamma), shape: (d_model,)
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

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

        # Upcast to float32 for numerical stability
        x_float = x.to(torch.float32)

        # Compute RMS over the last dimension
        # Shape: (B, T, 1)
        # rms = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(
            torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps
        )

        # Normalize
        x_norm = x_float / rms

        # Apply learnable scale (broadcast over batch & seq)
        out = x_norm * self.weight

        # Cast back to original dtype (e.g., float16 / bfloat16)
        return out.to(orig_dtype)
    

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
        device (optional): Device on which parameters are allocated.
        dtype (optional): Data type of the parameters.
    """

    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            # Common SwiGLU sizing rule used in LLaMA-style models
            d_ff = int((d_model * 8 / 3) // 64) * 64

        # # Compute d_ff ≈ 8/3 * d_model
        # d_ff = int((8 * d_model) / 3)

        # # Round up to nearest multiple of 64
        # d_ff = 64 * (d_ff // 64)

        # Sigmoid Linear Unit
        self.SiLU = SiLU()

        # Projection for the gated (SiLU) branch
        self.w1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)

        # Projection for the linear (gate) branch
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)

        # Output projection back to model dimension
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

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
        # Gated branch with SiLU activation
        gated = self.SiLU(self.w1(x))

        # Linear gate branch
        gate = self.w3(x)

        # Element-wise gating and projection back to d_model
        return self.w2(gate * gated)
   
    

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
        """
        Apply the SiLU activation function.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
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
    https://www.youtube.com/watch?v=hCzJo4ui1P8

    Args
    ----
    theta : float
        Base frequency constant (Θ in the paper).
    d_k : int
        Dimension of the query/key vectors (must be even).
    max_seq_len : int
        Maximum supported sequence length.
    device : torch.device | None
        Device on which buffers will be allocated.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RotaryPositionalEmbedding.")

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Create frequency scaling factors for each dimension pair.
        # Shape: (d_k // 2,)
        #
        # inv_freq[k] = 1 / (theta^(2k/d_k))
        #
        # This corresponds to the denominator term in theta_i,k.
        k = torch.arange(0, d_k // 2, device=device)
        inv_freq = 1.0 / (theta ** (2 * k / d_k))

        # Token positions (0 ... max_seq_len-1)
        positions = torch.arange(max_seq_len, device=device)

        # Compute rotation angles:
        # angles[i, k] = i * inv_freq[k]
        # Shape: (max_seq_len, d_k // 2)
        # angles = torch.outer(positions, inv_freq)
        angles = einsum(positions, inv_freq, 'i, j -> i j')

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
            raise ValueError(
                f"Last dimension of x must be {self.d_k}, got {x.shape[-1]}"
            )

        # Retrieve cos/sin values for the given token positions.
        # Resulting shape: (..., seq_len, d_k // 2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # Split even and odd dimensions:
        # x_even[..., k] = x[..., 2k]
        # x_odd[..., k]  = x[..., 2k+1]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply rotation:
        # x_even' = x_even * cos - x_odd * sin
        # x_odd'  = x_even * sin + x_odd * cos
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # Interleave rotated pairs back into original dimension layout.
        # Stack last dim as (..., seq_len, d_k/2, 2) then flatten.
        x_out = torch.stack((x_rot_even, x_rot_odd), dim=-1)
        x_out = x_out.flatten(-2)

        return x_out

    
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
        device:
            Optional device for parameter initialization.
        dtype:
            Optional data type for parameter initialization.
    """
    def __init__(self, d_model: int, num_heads: int, positional_embedding_layer=None, device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Projections (3 total, as in Vaswani et al.)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Output projection
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Optional RoPE module
        self.positional_embedding_layer = positional_embedding_layer

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None, past_k: Optional[torch.Tensor] = None, past_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute causal multi-head self-attention over an input sequence.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, seq_len, d_model).
            token_positions (Optional[torch.Tensor]):
                Tensor of shape (seq_len,) specifying the positional indices of
                each token in the sequence. Required when a rotary positional
                embedding layer is used. If omitted, positions are assumed to be
                [0, 1, ..., seq_len - 1].

        Returns:
            torch.Tensor:
                Output tensor of shape (batch_size, seq_len, d_model), where each
                token representation is updated by attending to all previous
                tokens in the sequence (including itself).
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V 
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 2. Split into heads 
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_head)
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)
        
        # 3. Treat heads as batch dimension for computational effeciency fot matmul 
        Q = rearrange(Q, "b h s d -> (b h) s d")
        K = rearrange(K, "b h s d -> (b h) s d")
        V = rearrange(V, "b h s d -> (b h) s d")
        
        # 4. Apply RoPE to Q and K (if provided) 
        if self.positional_embedding_layer is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.positional_embedding_layer(Q, token_positions)
            K = self.positional_embedding_layer(K, token_positions)

        # Causal mask 
        # causal_mask = torch.tril(
        #     torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        # )
        past_len = 0 if past_k is None else past_k.shape[1]
        total_len = past_len + seq_len

        causal_mask = torch.tril(
            torch.ones(seq_len, total_len, device=x.device),
            diagonal=past_len
        )

        # Concatenate past KV if provided
        if past_k is not None and past_v is not None:
            K = torch.cat([past_k, K], dim=1)
            V = torch.cat([past_v, V], dim=1)

        # Scaled dot-product attention 
        attn = self.scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Restore head structure 
        attn = rearrange(
            attn, "(b h) s d -> b s (h d)", b=batch_size, h=self.num_heads
        )

        # Output projection 
        # return self.output_proj(attn)
    
        return self.output_proj(attn), (K, V)
            

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
        scores = einsum(queries, keys,"... q d, ... k d -> ... q k") / math.sqrt(d_k)
        
        # Apply mask before softmax
        if mask is not None:
            # Use a very large negative number so masked positions become 0 after softmax
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Normalize scores across the key sequence length
        weights = utils.softmax(scores, dim=-1)
        
        # Apply weights to values
        # Equation: (batch, q, k) x (batch, k, v) -> (batch, q, v)
        return einsum(weights, values, "... q k, ... k v -> ... q v")
    


class TransformerBlock(nn.Module):
    """
    A transfomer block that contains two 'sublayers', one for the multihead self attention, and another for the feed-forward network. 
    In each sublayer, first perform RMSNorm, then the main operation (MHA/FF), finally adding in the residual connection
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, positional_embedding_layer: Optional[torch.Tensor] = None): 
        """
        d_model: int, Dimensionality of the Transformer block inputs.
        num_heads: int, Number of heads to use in multi-head self-attention. 
        d_ff: int, Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.attn = Multihead_self_attention(d_model=d_model, num_heads=num_heads, positional_embedding_layer=positional_embedding_layer)
        self.ffn = Positionwise_FeedForward(d_model=d_model, d_ff=d_ff)
    
    def forward(self, x: torch.Tensor, past_k=None, past_v=None) -> torch.Tensor:
        """
        x dim (batch_size, seq_len, d_model)
        """

        ## token_positions dim: (batch_size num_heads seq_len)
        seq_len = x.shape[-2]
        # token_positions = rearrange(torch.arange(seq_len), "seq_len -> 1 1 seq_len")
        # token_positions = torch.arange(seq_len, device=x.device)
        past_len = 0 if past_k is None else past_k.shape[1]
        token_positions = torch.arange(
            past_len, past_len + seq_len, device=x.device
        )

        ## out_1 dim: (batch_size, seq_len, d_model)
        attn_out, kv = self.attn(self.ln1(x), token_positions, past_k, past_v)
        out_1 = x + attn_out

        ## out_2 dim: (batch_size, seq_len, d_model)
        out_2 = out_1 + self.ffn(self.ln2(out_1))

        return out_2, kv


class TransformerLM(nn.Module):
    """
    Full language model. 
    """
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float = 10000.0):
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
        rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_model//num_heads, max_seq_len=context_length)

        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(f"{i}", TransformerBlock(d_model, num_heads, d_ff, positional_embedding_layer=rope))
        
        print('vocab_size:', vocab_size, ' context_length:',context_length,' num_layers:',num_layers,' d_model:',d_model,' num_heads',num_heads,' d_ff',d_ff)
        
    def forward(self, x: torch.Tensor, past_kvs=None):
        """
        x: Int[Tensor, "batch_size sequence_length"]
        output: Int[Tensor, "batch_size sequence_length vocab_size"], unnormalized prediction scores
        """
        
        ## embedding -> transformer blocks
        # out = self.layers(self.token_embeddings(x))
        out = self.token_embeddings(x)
        new_past_kvs = []
        for i, layer in enumerate(self.layers):
            if past_kvs is None:
                past_k, past_v = None, None
            else:
                past = past_kvs[i]
                if past is None:
                    past_k, past_v = None, None
                else:
                    past_k, past_v = past
            out, kv = layer(out, past_k=past_k, past_v=past_v)
            new_past_kvs.append(kv)

        ## norm, linear
        out_score = self.lm_head(self.ln_final(out))

        return out_score, new_past_kvs