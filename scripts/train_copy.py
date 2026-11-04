
import torch
from torch import nn
from pathlib import Path

from transformer.model import Transformer


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 32


def make_copy_batch(
    batch_size: int,
    max_tokens: int,
    device: torch.device,
):
    # 内容トークンの系列長: 1～max_tokens
    lengths = torch.randint(
        1, max_tokens + 1, (batch_size,), device=device
    )

    # 通常トークンは3以上
    content = torch.randint(
        3, VOCAB_SIZE,
        (batch_size, max_tokens),
        device=device,
    )

    positions = torch.arange(
        max_tokens, device=device
    ).unsqueeze(0)

    valid = positions < lengths.unsqueeze(1)

    # Source: [content..., EOS, PAD...]
    src = torch.full(
        (batch_size, max_tokens + 1),
        PAD_ID, dtype=torch.long, device=device,
    )

    src[:, :max_tokens] = torch.where(
        valid,
        content,
        torch.full_like(content, PAD_ID),
    )

    src.scatter_(
        1, lengths.unsqueeze(1), EOS_ID
    )

    # Decoder input: [BOS, content..., PAD...]
    tgt_in = torch.full_like(src, PAD_ID)
    tgt_in[:, 0] = BOS_ID
    tgt_in[:, 1:] = torch.where(
        valid,
        content,
        torch.full_like(content, PAD_ID),
    )

    # Labels: [content..., EOS, PAD...]
    labels = src.clone()

    return src, tgt_in, labels


def create_model(device: torch.device):
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=32,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        dropout=0.1,
        src_pad_id=PAD_ID,
        tgt_pad_id=PAD_ID,
        max_len=64,
    )
    return model.to(device)


def train():
    torch.manual_seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model = create_model(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=PAD_ID
    )

    model.train()

    for step in range(1, 5001):
        src, tgt_in, labels = make_copy_batch(
            batch_size=64,
            max_tokens=8,
            device=device,
        )

        # Forward
        logits = model(src, tgt_in)

        # logits: (B, N, V)
        # labels: (B, N)
        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1),
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        # Parameter update
        optimizer.step()

        if step % 500 == 0:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                valid = labels != PAD_ID

                accuracy = (
                    ((predictions == labels) & valid).sum()
                    / valid.sum()
                ).item()

            print(
                f"Step {step:4d} | "
                f"Loss {loss.item():.4f} | "
                f"Token accuracy {accuracy:.3f}"
            )

    return model, device


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_new_tokens: int = 20,
):
    model.eval()

    if src.shape[0] != 1:
        raise ValueError("This function supports batch size 1.")

    generated = torch.full(
        (1, 1),
        BOS_ID,
        dtype=torch.long,
        device=src.device,
    )

    for _ in range(max_new_tokens):
        logits = model(src, generated)

        # Predict the next token
        next_token = logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

        if next_token.item() == EOS_ID:
            break

    return generated


@torch.no_grad()
def inspect_predictions(model, src):
    model.eval()

    # Copy Taskなので、Sourceから正解系列を構成できる
    tgt_in = torch.cat(
        [
            torch.full(
                (1, 1), BOS_ID,
                dtype=torch.long,
                device=src.device,
            ),
            src[:, :-1],
        ],
        dim=1,
    )

    logits = model(src, tgt_in)

    teacher_predictions = logits.argmax(dim=-1)
    greedy_predictions = greedy_decode(
        model, src, max_new_tokens=src.shape[1] + 2
    )

    print("Source:        ", src[0].tolist())
    print("Teacher forced:", teacher_predictions[0].tolist())
    print("Greedy:        ", greedy_predictions[0].tolist())
    print("Expected:      ", src[0].tolist())


@torch.no_grad()
def evaluate(model, device, num_batches=100):
    model.eval()

    total_correct = 0
    total_tokens = 0
    exact_matches = 0
    total_sequences = 0

    for _ in range(num_batches):
        src, tgt_in, labels = make_copy_batch(
            batch_size=64,
            max_tokens=8,
            device=device,
        )

        logits = model(src, tgt_in)
        pred = logits.argmax(dim=-1)

        valid = labels != PAD_ID
        correct = (pred == labels) & valid

        total_correct += correct.sum().item()
        total_tokens += valid.sum().item()

        # 1つの系列の有効位置がすべて正解か
        sequence_correct = (
            (pred == labels) | ~valid
        ).all(dim=-1)

        exact_matches += sequence_correct.sum().item()
        total_sequences += labels.shape[0]

    token_accuracy = total_correct / total_tokens
    sequence_accuracy = exact_matches / total_sequences

    print(f"Validation Token Accuracy: {token_accuracy:.4f}")
    print(f"Validation Exact Match:    {sequence_accuracy:.4f}")

    return token_accuracy, sequence_accuracy


if __name__ == "__main__":
    model, device = train()

    checkpoint_path = Path("checkpoints/copy_transformer.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step": 5000,
            "config": {
                "src_vocab_size": VOCAB_SIZE,
                "tgt_vocab_size": VOCAB_SIZE,
                "d_model": 32,
                "num_heads": 4,
                "d_ff": 128,
                "num_layers": 2,
                "dropout": 0.1,
                "src_pad_id": PAD_ID,
                "tgt_pad_id": PAD_ID,
                "max_len": 64,
            },
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint to {checkpoint_path}")

    src = torch.tensor(
        [[7, 15, 22, 9, EOS_ID]],
        dtype=torch.long,
        device=device,
    )

    predicted = greedy_decode(model, src)
    inspect_predictions(model, src)
    evaluate(model, device)