# open-llm

A lightweight, modular, and optimized implementation of a modern **GPT-style Large Language Model (LLM)** built from scratch in PyTorch. 

`open-llm` includes a complete pipeline for custom Byte-Pair Encoding (BPE) tokenization, memory-mapped streaming datasets, custom optimizer implementations, mixed-precision training, autoregressive text generation, and empirical attention/memory benchmarks.

---

## Key Features & Architecture

### Model Architecture (`model/`)
- **Pre-LayerNorm Architecture**: Uses root-mean-square normalization (**RMSNorm**) prior to attention and feed-forward sublayers for enhanced training stability.
- **Rotary Position Embeddings (RoPE)**: Position-dependent 2D vector rotations applied to query and key states (`RotaryPositionalEmbedding`).
- **Causal Multi-Head Self-Attention**: Efficient matrix multiplication using scaled dot-product attention with causal masking.
- **SwiGLU Activation**: Position-wise Feed-Forward Networks utilizing Swish Gated Linear Units (`Positionwise_FeedForward` with `SiLU`).
- **Custom Initialization**: Truncated normal parameter initialization across projection and embedding weights.

### Byte-Pair Encoding Tokenizer (`tokenizer/`)
- **Custom BPE Algorithm**: Fast pair-frequency calculation and iterative merging with regex pre-tokenization (`PAT`).
- **Special Token Support**: Full handling of custom control tokens such as `<|endoftext|>`.
- **Numpy Array Serialization**: Fast encoding pipeline (`encode_dataset`) storing token IDs as compact `uint16` binary files (`.npy`).
- **Memory-Mapped Data Loader**: `get_batch` utility powered by `np.memmap` for streaming training from multi-gigabyte datasets with minimal RAM consumption.

### Optimizer & Training Pipeline (`model/optimizer.py`, `model/train.py`)
- **Custom AdamW Optimizer**: Decoupled weight decay implementation following Loshchilov & Hutter.
- **Cosine Learning Rate Schedule**: Warmup phase (`T_w`) followed by cosine annealing (`T_c`) down to minimum learning rate.
- **Mixed Precision Support**: Integrates `torch.cuda.amp` / `autocast` for faster throughput and lower VRAM consumption.
- **Checkpointing & Resuming**: Saving/restoring optimizer state and model parameters (`serialization.py`).

### Autoregressive Inference (`model/gen_text.py`)
- **Text Generation Engine**: Supports nucleus (Top-$p$) sampling and temperature scaling.
- **KV-Cache Optimization**: Efficient incremental generation avoiding re-computation of key/value states.

---

## Benchmarks & Performance Analysis (`experiments/`)

The repository includes detailed empirical benchmarks comparing attention mechanisms, memory scaling, and precision modes in `experiments/benchmark.ipynb`.

### 1. Flash Attention 2 vs. Manual SDPA Speed Comparison
Comparing forward and backward pass execution time across different sequence lengths ($256$ to $16,384$) and embedding dimensions ($d_{\text{model}} \in \{16, 32, 64, 128\}$).

![Flash Attention Speed & Memory](experiments/flash%20attention%20speed%20and%20mem.png)

### 2. Peak Memory Usage
Peak VRAM consumption analysis during attention forward and backward operations, demonstrating sub-quadratic memory scaling with Flash Attention.

![Peak Memory Usage](experiments/max%20mem.png)

### 3. Mixed Precision Training Latency
Step latency comparison across model benchmark modes on a 6-layer Transformer (`BasicsTransformerLM`).

![Mixed Precision Benchmark](experiments/mixed%20precision.png)

#### Benchmark Performance Summary Table

| Benchmark Mode | Average Time per Step (s) | Speedup / Notes |
| :--- | :---: | :--- |
| **Forward Only** | `0.0216 s` | Inference / Eval forward baseline |
| **Forward + Backward** | `0.0279 s` | Gradient computation pass |
| **Full Train (No MP)** | `0.0301 s` | FP32 standard training step |
| **Full Train (With MP)** | `0.0246 s` | **~18.3% Speedup** via Mixed Precision (`autocast`) |

---

## Repository Layout

```
open-llm/
├── model/                  # Core Transformer LM architecture & training logic
│   ├── model.py            # Embedding, RMSNorm, RoPE, SwiGLU, Multihead Attention, TransformerLM
│   ├── train.py            # CLI training script with validation & checkpointing
│   ├── optimizer.py        # Custom AdamW optimizer & Cosine LR schedule with warmup
│   ├── gen_text.py         # Autoregressive text generation with Top-p sampling
│   ├── serialization.py    # Save/load model and optimizer state checkpoints
│   └── utils.py            # Math utilities and custom activation functions
├── tokenizer/              # BPE Tokenizer implementation & data processing
│   ├── tokenizer.py        # BPE Tokenizer class (encode/decode, special tokens, cache)
│   ├── train_bpe.py        # BPE vocabulary trainer
│   └── data_loader.py      # Memory-mapped dataset loader (get_batch)
├── experiments/            # Benchmark scripts, plots, and analysis
│   ├── benchmark.ipynb     # Interactive Jupyter notebook for benchmarks
│   ├── flash attention speed and mem.png  # Forward/backward speed comparison plot
│   ├── max mem.png         # Peak memory usage comparison plot
│   └── mixed precision.png # Training step time comparison plot
├── data/                   # Tokenizer vocabulary files and tokenized dataset buffers
├── checkpoints/            # Directory for saved model checkpoints
├── pyproject.toml          # Project metadata and dependencies (`uv` / `pip`)
└── README.md               # Project documentation
```

---

## Getting Started

### 1. Prerequisites & Installation

Ensure you have Python `>=3.11` installed. You can install dependencies using [`uv`](https://github.com/astral-sh/uv) or standard `pip`:

```bash
# Clone the repository
git clone https://github.com/jmanu127/open-llm.git
cd open-llm

# Install dependencies using uv
uv sync
```

### 2. Prepare Tokenizer & Dataset

Train a BPE tokenizer or encode raw text datasets into binary format:

```bash
# Encode text corpus into binary .npy token arrays
python -m tokenizer.tokenizer
```

### 3. Train the Model

Launch model training with configurable model dimensions, learning rate, and batch size:

```bash
python model/train.py \
  --train_data data/train_tokens.npy \
  --val_data data/val_tokens.npy \
  --vocab_size 10000 \
  --context_length 256 \
  --n_layers 6 \
  --n_heads 8 \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 32 \
  --lr 3e-4 \
  --max_iters 50000 \
  --device mps
```

### 4. Text Generation / Inference

Generate text from a prompt using a saved checkpoint:

```bash
python model/gen_text.py
```

---

## Implementation Status & Roadmap

### Completed Features
- [x] GPT-style Pre-LN Transformer architecture
- [x] RMSNorm (Root Mean Square Layer Normalization)
- [x] Rotary Position Embeddings (RoPE)
- [x] SwiGLU feed-forward networks
- [x] Custom Byte-Pair Encoding (BPE) tokenizer
- [x] Memory-mapped (`np.memmap`) streaming data loader
- [x] Custom AdamW optimizer with decoupled weight decay
- [x] Cosine LR scheduler with linear warmup
- [x] Mixed precision training support (`autocast`)
- [x] KV-cache inference & nucleus (Top-$p$) sampling
- [x] Attention & memory benchmarking suite (Flash Attention 2 comparison)

### 🔮 Roadmap & Future Scope
- [ ] Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)
- [ ] Custom CUDA kernels for fused operations
- [ ] Distributed Training (DDP / FSDP / Megatron-LM style tensor parallelism)
- [ ] Parameter-efficient fine-tuning (LoRA / QLoRA)
- [ ] Alignment (RLHF / DPO)
- [ ] Mixture-of-Experts (MoE) layer integration
- [ ] Function calling & tool usage capabilities

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
