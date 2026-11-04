
import copy
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn

from transformer.model import Transformer


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 32

D_MODEL = 32
STEPS = 5000
BATCH_SIZE = 64
MAX_TOKENS = 8
LR = 3e-4
SEED = 42


def create_model():
    return Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        dropout=0.1,
        src_pad_id=PAD_ID,
        tgt_pad_id=PAD_ID,
        max_len=64,
    )


def create_batch(generator, batch_size, fixed_length=False):
    """Create a Copy Task batch on CPU."""

    if fixed_length:
        lengths = torch.full(
            (batch_size,), MAX_TOKENS, dtype=torch.long
        )
    else:
        lengths = torch.randint(
            1, MAX_TOKENS + 1,
            (batch_size,),
            generator=generator,
        )

    content = torch.randint(
        3, VOCAB_SIZE,
        (batch_size, MAX_TOKENS),
        generator=generator,
    )

    positions = torch.arange(MAX_TOKENS)[None, :]
    valid = positions < lengths[:, None]

    src = torch.full(
        (batch_size, MAX_TOKENS + 1),
        PAD_ID,
        dtype=torch.long,
    )

    src[:, :MAX_TOKENS] = torch.where(
        valid, content, PAD_ID
    )
    src.scatter_(1, lengths[:, None], EOS_ID)

    tgt_in = torch.full_like(src, PAD_ID)
    tgt_in[:, 0] = BOS_ID
    tgt_in[:, 1:] = torch.where(
        valid, content, PAD_ID
    )

    labels = src.clone()
    return src, tgt_in, labels


def modify_embedding_initialization(model):
    """Change N(0,1) to N(0,1/D)."""

    with torch.no_grad():
        for embedding in (
            model.encoder.embedding,
            model.decoder.embedding,
        ):
            embedding.token_embedding.weight.div_(
                math.sqrt(D_MODEL)
            )


@torch.no_grad()
def measure_embedding_rms(model):
    results = []

    for embedding in (
        model.encoder.embedding,
        model.decoder.embedding,
    ):
        E = embedding.token_embedding.weight
        scaled_E = math.sqrt(D_MODEL) * E
        results.append(
            scaled_E.square().mean().sqrt().item()
        )

    return results


@torch.no_grad()
def evaluate(model, validation_batches, device):
    model.eval()

    content_correct = 0
    content_total = 0
    exact_correct = 0
    sequence_total = 0

    for src, tgt_in, labels in validation_batches:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        labels = labels.to(device)

        logits = model(src, tgt_in)
        pred = logits.argmax(dim=-1)

        content_mask = labels >= 3

        content_correct += (
            ((pred == labels) & content_mask)
            .sum().item()
        )
        content_total += content_mask.sum().item()

        # Length 8 means no PAD; EOS is included.
        exact_correct += (
            (pred == labels).all(dim=-1)
            .sum().item()
        )
        sequence_total += labels.shape[0]

    return {
        "content_accuracy": content_correct / content_total,
        "exact_match": exact_correct / sequence_total,
    }


def train_condition(
    model,
    device,
    name,
    validation_batches,
):
    model = model.to(device)

    # Same dropout RNG seed for each condition.
    torch.manual_seed(SEED)

    # Independent data RNG, identical for A and B.
    data_generator = torch.Generator().manual_seed(12345)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    history = []

    for step in range(1, STEPS + 1):
        model.train()

        src, tgt_in, labels = create_batch(
            data_generator, BATCH_SIZE
        )

        src = src.to(device)
        tgt_in = tgt_in.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(src, tgt_in)

        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        if step % 500 == 0:
            metrics = evaluate(
                model, validation_batches, device
            )

            record = {
                "condition": name,
                "step": step,
                "train_loss": loss.item(),
                **metrics,
            }
            history.append(record)

            print(
                f"{name} | Step {step:4d} | "
                f"Loss {loss.item():.4f} | "
                f"Content {metrics['content_accuracy']:.3f} | "
                f"Exact {metrics['exact_match']:.3f}"
            )

    return model, history


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    torch.manual_seed(SEED)
    initial_model = create_model()

    # Model A: PyTorch default initialization
    model_a = copy.deepcopy(initial_model)

    # Model B: Same embeddings, rescaled by 1/sqrt(D)
    model_b = copy.deepcopy(initial_model)
    modify_embedding_initialization(model_b)

    for name, model in [
        ("A_default", model_a),
        ("B_scaled_init", model_b),
    ]:
        src_rms, tgt_rms = measure_embedding_rms(model)
        print(
            f"{name} | "
            f"Source RMS={src_rms:.4f}, "
            f"Target RMS={tgt_rms:.4f}"
        )

    # Fixed validation dataset: length exactly 8
    val_generator = torch.Generator().manual_seed(2026)
    validation_batches = [
        create_batch(
            val_generator,
            BATCH_SIZE,
            fixed_length=True,
        )
        for _ in range(20)
    ]

    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    history = []

    for name, model in [
        ("A_default", model_a),
        ("B_scaled_init", model_b),
    ]:
        print(f"\n=== Training {name} ===")

        model, records = train_condition(
            model,
            device,
            name,
            validation_batches,
        )
        history.extend(records)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "condition": name,
                "seed": SEED,
                "steps": STEPS,
            },
            f"checkpoints/{name}.pt",
        )

    with open(
        "results/embedding_initialization.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "step",
                "train_loss",
                "content_accuracy",
                "exact_match",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
