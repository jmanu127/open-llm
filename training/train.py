import argparse
import math
import numpy as np
import torch
import torch.nn as nn

from model import TransformerLM
from optimizer import AdamW
from data_loader import get_batch
from serialization import save_checkpoint, load_checkpoint


def evaluate(model, data, batch_size, context_length, device, num_batches=20):
    """Estimate mean loss over several batches."""
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, batch_size, context_length, device)
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main(args):

    device = torch.device(args.device)

    # ----------------------------
    # Load datasets with memmap
    # ----------------------------
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    # ----------------------------
    # Build model
    # ----------------------------
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

    # ----------------------------
    # Resume from checkpoint
    # ----------------------------
    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed training from iteration {start_iter}")

    # ----------------------------
    # Training loop
    # ----------------------------
    for iteration in range(start_iter, args.max_iters):

        # ---- get batch ----
        x, y = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            args.device
        )

        # ---- forward ----
        logits, _ = model(x)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        # ---- backward ----
        loss.backward()

        # ---- optimizer ----
        optimizer.step()
        optimizer.zero_grad()

        # ---- logging ----
        if iteration % args.log_interval == 0:
            val_loss = evaluate(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                args.device
            )

            print(
                f"iter {iteration} | "
                f"train loss {loss.item():.4f} | "
                f"val loss {val_loss:.4f} | "
                f"ppl {math.exp(val_loss):.2f}"
            )

        # ---- checkpointing ----
        if iteration % args.ckpt_interval == 0 and iteration > 0:
            save_checkpoint(
                model,
                optimizer,
                iteration,
                args.checkpoint_path
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)

    # model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)

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