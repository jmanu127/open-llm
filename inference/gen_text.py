import torch
import torch.nn.functional as F
import math


from model import TransformerLM
from optimizer import AdamW
from data_loader import get_batch
from serialization import save_checkpoint, load_checkpoint
from tokenizer import Tokenizer

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_p=0.9,
    device="mps"
):
    model.eval()
    device = torch.device(device)
    model = model.to(device)

    if isinstance(prompt, str):
        tokens = tokenizer.encode(prompt)
    else:
        tokens = prompt

    generated = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Run prompt through model first
    logits, past_kvs = model(generated)

    for _ in range(max_new_tokens):

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        # Top-p sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            probs = torch.zeros_like(probs)
            probs.scatter_(1, sorted_indices, sorted_probs)

        next_token = torch.multinomial(probs, 1)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.bytes_to_id[b"<|endoftext|>"]:
            break

        logits, past_kvs = model(next_token, past_kvs=past_kvs)

        # print(tokenizer.decode([next_token.item()]), end="", flush=True)

    return tokenizer.decode(generated[0].tolist())


if __name__ == "__main__":
    # Load model (assume vocab_size=32000, etc.)
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        rope_theta=10000,
    ).to("mps")  # or "cpu"

    # Load a trained checkpoint if available
    load_checkpoint("checkpoint.pt", model, None)
    tokenizer = Tokenizer.from_files(
        "data/vocab.json",
        "data/merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    prompt = "Once upon a time"
    output = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=500,
        temperature=0.9,
        top_p=0.9,
        device="mps",
    )

    print(output)