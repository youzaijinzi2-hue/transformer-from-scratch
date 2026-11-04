
import torch
import matplotlib.pyplot as plt

from transformer.model import Transformer
from train_copy import (
    make_copy_batch,
    PAD_ID,
    BOS_ID,
    EOS_ID,
)
from transformer.masking import (
    make_padding_mask,
    make_target_mask,
)



def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = Transformer(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def greedy_decode_batch(
    model,
    src,
    bos_id=1,
    eos_id=2,
    pad_id=0,
    max_new_tokens=10,
):
    batch_size = src.shape[0]

    generated = torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=src.device,
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=src.device,
    )

    for _ in range(max_new_tokens):
        logits = model(src, generated)
        next_token = logits[:, -1, :].argmax(dim=-1)

        # Finished sequences receive PAD thereafter
        next_token = torch.where(
            finished,
            torch.full_like(next_token, pad_id),
            next_token,
        )

        generated = torch.cat(
            [generated, next_token[:, None]],
            dim=1,
        )

        finished |= next_token == eos_id

        if finished.all():
            break

    return generated


@torch.no_grad()
def evaluate_by_length(
    model,
    device,
    num_batches=10,
    batch_size=32,
    max_tokens=8,
):
    model.eval()

    stats = {
        n: {
            "tokens_correct": 0,
            "tokens_total": 0,
            "tf_exact": 0,
            "greedy_exact": 0,
            "total": 0,
        }
        for n in range(1, max_tokens + 1)
    }

    for _ in range(num_batches):
        src, tgt_in, labels = make_copy_batch(
            batch_size=batch_size,
            max_tokens=max_tokens,
            device=device,
        )

        # Teacher Forcing
        logits = model(src, tgt_in)
        predictions = logits.argmax(dim=-1)

        # Autoregressive Greedy Decoding
        generated = greedy_decode_batch(
            model,
            src,
            max_new_tokens=max_tokens + 2,
        )

        valid = labels != 0
        lengths = valid.sum(dim=-1) - 1

        tf_correct = (
            (predictions == labels) | ~valid
        ).all(dim=-1)

        for b in range(batch_size):
            n = int(lengths[b].item())
            result = stats[n]

            result["total"] += 1
            result["tf_exact"] += int(tf_correct[b].item())

            result["tokens_correct"] += int(
                ((predictions[b] == labels[b]) & valid[b])
                .sum().item()
            )
            result["tokens_total"] += int(valid[b].sum().item())

            # Ignore the initial BOS and trailing PADs
            actual = generated[b, 1:].tolist()
            expected = src[b].tolist()

            def trim(sequence):
                if 2 in sequence:
                    return sequence[:sequence.index(2) + 1]
                return [x for x in sequence if x != 0]

            result["greedy_exact"] += int(
                trim(actual) == trim(expected)
            )

    print("Length | Token Acc | TF Exact | Greedy Exact")
    print("-" * 48)

    for n, result in stats.items():
        if result["total"] == 0:
            continue

        token_acc = (
            result["tokens_correct"] / result["tokens_total"]
        )
        tf_acc = result["tf_exact"] / result["total"]
        greedy_acc = result["greedy_exact"] / result["total"]

        print(
            f"{n:6d} | "
            f"{token_acc:9.3f} | "
            f"{tf_acc:8.3f} | "
            f"{greedy_acc:12.3f}"
        )


@torch.no_grad()
def get_cross_attention(model, src, tgt_in):
    model.eval()

    source_mask = make_padding_mask(
        src, pad_id=model.src_pad_id
    )

    target_mask = make_target_mask(
        tgt_in, pad_id=model.tgt_pad_id
    )

    # Encoder
    memory, _ = model.encoder(
        src, mask=source_mask
    )

    # Decoder
    _, self_maps, cross_maps = model.decoder(
        tgt_in,
        memory,
        target_mask,
        source_mask,
    )

    return self_maps, cross_maps


@torch.no_grad()
def plot_cross_attention(model, src, tgt_in, layer=-1):
    _, cross_maps = get_cross_attention(
        model, src, tgt_in
    )

    # Select one decoder layer and the first sample
    A = cross_maps[layer][0].detach().cpu()

    num_heads = A.shape[0]

    for h in range(num_heads):
        fig, ax = plt.subplots(figsize=(6, 5))

        image = ax.imshow(
            A[h].numpy(),
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
        )

        ax.set_xlabel("Encoder key position")
        ax.set_ylabel("Decoder query position")
        ax.set_title(
            f"Cross-Attention | Layer {layer}, Head {h}"
        )

        fig.colorbar(image, ax=ax, label="Attention weight")
        fig.tight_layout()

        fig.savefig(
            f"cross_attention_head_{h}.png",
            dpi=200,
        )

        plt.close(fig)


import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def plot_accuracy_by_position(
    model, device,
    num_batches=100,
    max_tokens=8,
):
    model.eval()

    correct = torch.zeros(
        max_tokens, max_tokens + 1,
        dtype=torch.float64,
    )
    counts = torch.zeros_like(correct)

    for _ in range(num_batches):
        src, tgt_in, labels = make_copy_batch(
            batch_size=64,
            max_tokens=max_tokens,
            device=device,
        )

        pred = model(src, tgt_in).argmax(dim=-1)

        lengths = (labels != PAD_ID).sum(dim=-1) - 1

        for n in range(1, max_tokens + 1):
            selected = lengths == n

            if selected.any():
                matches = (
                    pred[selected, :n + 1]
                    == labels[selected, :n + 1]
                )

                correct[n - 1, :n + 1] += (
                    matches.sum(dim=0).cpu()
                )
                counts[n - 1, :n + 1] += selected.sum().item()

    accuracy = correct / counts.clamp_min(1)
    accuracy[counts == 0] = float("nan")

    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        accuracy.numpy(),
        vmin=0, vmax=1,
        aspect="auto",
        cmap="viridis",
    )

    ax.set_xlabel("Prediction position (EOS included)")
    ax.set_ylabel("Content sequence length")
    ax.set_yticks(range(max_tokens))
    ax.set_yticklabels(range(1, max_tokens + 1))
    ax.set_xticks(range(max_tokens + 1))

    fig.colorbar(im, ax=ax, label="Token accuracy")
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def evaluate_content_and_eos(
    model,
    device,
    num_batches=100,
):
    model.eval()

    content_correct = 0
    content_total = 0
    eos_correct = 0
    eos_total = 0

    for _ in range(num_batches):
        src, tgt_in, labels = make_copy_batch(
            batch_size=64,
            max_tokens=8,
            device=device,
        )

        logits = model(src, tgt_in)
        pred = logits.argmax(dim=-1)

        content_mask = (
            (labels != PAD_ID) &
            (labels != EOS_ID)
        )
        eos_mask = labels == EOS_ID

        content_correct += (
            ((pred == labels) & content_mask).sum().item()
        )
        content_total += content_mask.sum().item()

        eos_correct += (
            ((pred == labels) & eos_mask).sum().item()
        )
        eos_total += eos_mask.sum().item()

    print(
        "Content Accuracy:",
        content_correct / content_total,
    )
    print(
        "EOS Accuracy:",
        eos_correct / eos_total,
    )


@torch.no_grad()
def inspect_embedding_scale(model, src):
    embedding = model.encoder.embedding

    E = embedding.token_embedding(src)
    E = E * (embedding.d_model ** 0.5)

    N = src.shape[1]
    PE = embedding.position.pe[:, :N, :]
    PE = PE.to(device=E.device, dtype=E.dtype)

    # Exclude PAD positions
    valid = src != model.src_pad_id
    E_valid = E[valid]
    PE_valid = PE.expand_as(E)[valid]

    rms_E = E_valid.square().mean().sqrt().item()
    rms_PE = PE_valid.square().mean().sqrt().item()

    print(f"Embedding RMS: {rms_E:.4f}")
    print(f"PE RMS:        {rms_PE:.4f}")
    print(f"Ratio E/PE:    {rms_E / rms_PE:.4f}")

if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # 1. Load trained model
    model = load_model(
        "checkpoints/copy_transformer.pt",
        device,
    )

    # 2. Evaluate accuracy by sequence length
    print("\n=== Accuracy by Sequence Length ===")
    evaluate_by_length(
        model,
        device,
        num_batches=100,
        batch_size=64,
        max_tokens=8,
    )

    # 3. Visualize accuracy by prediction position
    print("\n=== Accuracy by Position ===")
    plot_accuracy_by_position(
        model,
        device,
        num_batches=100,
        max_tokens=8,
    )

    # 4. Visualize cross-attention
    print("\n=== Cross-Attention ===")

    src = torch.tensor(
        [[7, 15, 22, 9, EOS_ID]],
        dtype=torch.long,
        device=device,
    )

    tgt_in = torch.tensor(
        [[BOS_ID, 7, 15, 22, 9]],
        dtype=torch.long,
        device=device,
    )

    plot_cross_attention(
        model,
        src,
        tgt_in,
        layer=-1,
    )

    evaluate_content_and_eos(model, device)
    inspect_embedding_scale(model, src)