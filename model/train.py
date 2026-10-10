import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import trackio
from data_loader import get_batch
from model import TransformerLM
from optimizer import AdamW
from serialization import load_checkpoint, save_checkpoint
from utils import cross_entropy, softmax, gradient_clipping


def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def compute_token_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float()
    return correct.mean().item()


def compute_entropy(logits):
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy.item()


def evaluate(model, data, batch_size, context_length, device, num_batches=20):
    """Estimate mean loss over several batches."""
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, batch_size, context_length, device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main(args):

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load datasets with memmap
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    
    # Build model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.n_layers,
        num_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=10000,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # print model
    # for name, param in model.named_parameters():
    #     print(name, param.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,}")
    print(f"{total_params / 1e6:.2f}M parameters")

    
    # Resume from checkpoint
    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed training from iteration {start_iter}")

    trackio.init(
        project="llm training",
        config={"iteration": args.max_iters, "learning_rate": args.lr, "batch_size": args.batch_size},
    )

    # Training loop
    tokens_per_iter = args.batch_size * args.context_length

    for iteration in range(start_iter, args.max_iters):
        # epoch estimate
        epoch = (iteration * tokens_per_iter) / len(train_data)

        # get batch
        x, y = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            args.device,
        )

        # forward
        logits = model(x)

        # loss = nn.functional.cross_entropy(
        #     logits.view(-1, logits.size(-1)),
        #     y.view(-1),
        # )
        loss = cross_entropy(logits, y)

        # metrics
        token_acc = compute_token_accuracy(logits, y)
        entropy = compute_entropy(logits)
        perplexity = torch.exp(loss).item()

        # backward
        loss.backward()

        # gradient norm
        grad_norm = compute_grad_norm(model)

        gradient_clipping(model.parameters(), args.max_norm)

        # optimizer
        optimizer.step()
        optimizer.zero_grad()

        # learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # logging
        trackio.log(
            {
                "iteration": iteration,
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_perplexity": perplexity,
                "learning_rate": current_lr,
                "gradient_norm": grad_norm,
                "entropy": entropy,
                "mean_token_accuracy": token_acc,
                
            }
        )

        if iteration % args.log_interval == 0:
            val_loss = evaluate(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                args.device,
            )

            ppl = math.exp(val_loss)

            print(
                f"iter {iteration} | "
                f"epoch {epoch:.3f} | "
                f"train loss {loss.item():.4f} | "
                f"val loss {val_loss:.4f} | "
                f"ppl {ppl:.2f} | "
                f"acc {token_acc:.4f} | "
                f"entropy {entropy:.4f} | "
                f"grad_norm {grad_norm:.4f} | "
                f"lr {current_lr:.2e}"
            )

            trackio.log(
                {
                    "val_loss": val_loss,
                    "Perplexity": ppl,
                }
            )

        # checkpointing
        if iteration % args.ckpt_interval == 0 and iteration > 0:
            save_checkpoint(
                model,
                optimizer,
                iteration,
                args.checkpoint_path,
            )

    trackio.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)

    # model
    parser.add_argument("--vocab_size", type=int, required=True, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)  # usually 768
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--max_norm", type=float, default=1.0)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=50000)

    # logging
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--ckpt_interval", type=int, default=1000)

    # checkpointing
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume", type=str, default=None)

    # device
    parser.add_argument("--device", type=str, default="mps")
    
    

    args = parser.parse_args()

    main(args)
