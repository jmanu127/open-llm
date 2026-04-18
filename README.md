# open-llm
Building an llm from scratch

This repo is a working progress.

## Implemented: 
GPT-style Transformer architecture
masked multi-head attention
RoPE embeddings
Pre-LayerNorm architecture
byte-pair encoding tokenizer
streaming training dataset pipeline
SwiGLU
RMSNorm

training pipeline with AdamW
gradient clipping
mixed precision training
cosine LR scheduling.
KV-cache inference optimization

## TODO:
trackio
Flash Attention
Grouped Query Attention and Multi-Query Attention
kernels
distributed training
parameter scaling
LoRA
finetuning
RLHF
Mixture-of-Experts (MoE)
Retrieval-Augmented Generation (RAG)
Tool use / function calling
