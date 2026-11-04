
import copy
import csv
import math
from pathlib import Path

import torch
from torch import nn

from transformer.model import Transformer
from compare_embedding_init import (
    create_model,
    create_batch,
    train_condition,
    BATCH_SIZE,
    SEED,
    D_MODEL,
    EOS_ID,
    BOS_ID,
)


def create_ablation_models(initial_model, d_model):
    root_d = math.sqrt(d_model)

    models = {}

    for condition in ["A", "B", "C", "D"]:
        model = copy.deepcopy(initial_model)

        for embedding in (
            model.encoder.embedding,
            model.decoder.embedding,
        ):
            if condition == "A":
                # X = sqrt(D) * E0 + PE
                embedding.embedding_scale = root_d
                embedding.positional_scale = 1.0

            elif condition == "B":
                # X = sqrt(D) * (E0 / sqrt(D)) + PE
                with torch.no_grad():
                    embedding.token_embedding.weight.div_(root_d)

                embedding.embedding_scale = root_d
                embedding.positional_scale = 1.0

            elif condition == "C":
                # X = E0 + PE
                embedding.embedding_scale = 1.0
                embedding.positional_scale = 1.0

            elif condition == "D":
                # X = sqrt(D) * (E0 + PE)
                embedding.embedding_scale = root_d
                embedding.positional_scale = root_d

        models[condition] = model

    return models

@torch.no_grad()
def verify_equivalence(model_b, model_c):
    model_b.eval()
    model_c.eval()

    src = torch.tensor([[7, 15, 22, 9, EOS_ID]])
    tgt = torch.tensor([[BOS_ID, 7, 15, 22, 9]])

    # Check embedding output
    for name, tokens in [
        ("encoder", src),
        ("decoder", tgt),
    ]:
        emb_b = getattr(model_b, name).embedding(tokens)
        emb_c = getattr(model_c, name).embedding(tokens)

        torch.testing.assert_close(
            emb_b, emb_c,
            rtol=1e-5,
            atol=1e-6,
        )

    # Check complete Transformer output
    logits_b = model_b(src, tgt)
    logits_c = model_c(src, tgt)

    torch.testing.assert_close(
        logits_b, logits_c,
        rtol=1e-5,
        atol=1e-6,
    )

    print("B and C produce equivalent initial outputs.")


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    torch.manual_seed(SEED)

    # Common initial parameters
    initial_model = create_model()

    # Generate four conditions
    models = create_ablation_models(
        initial_model,
        d_model=D_MODEL,
    )

    # Verify B and C are initially equivalent
    verify_equivalence(models["B"], models["C"])

    # Fixed validation dataset: content length = 8
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

    for condition, model in models.items():
        print(f"\n=== Training condition {condition} ===")

        model, records = train_condition(
            model,
            device,
            condition,
            validation_batches,
        )

        history.extend(records)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "condition": condition,
                "seed": SEED,
                "embedding_scale": (
                    model.encoder.embedding.embedding_scale
                ),
                "positional_scale": (
                    model.encoder.embedding.positional_scale
                ),
            },
            f"checkpoints/ablation_{condition}.pt",
        )

    with open(
        "results/embedding_ablation.csv",
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

    print("\nAblation study completed.")


if __name__ == "__main__":
    main()