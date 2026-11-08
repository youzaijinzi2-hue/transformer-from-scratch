# Transformer from Scratch

PyTorchでEncoder–Decoder型Transformerを構成要素から実装する学習用プロジェクトです。Attention、位置エンコーディング、マスク、正規化を実装し、整数トークン列をそのまま出力する**コピータスク**で学習・推論の動作を確認します。

## 実装内容

- Scaled Dot-Product Attention / Multi-Head Attention
- 正弦波による位置エンコーディングとトークン埋め込み
- Position-wise Feed Forward Network、Layer Normalization、残差接続
- Post-LN構成のEncoder / DecoderとCross-Attention
- Padding MaskとCausal Mask
- Teacher Forcingによる学習、自己回帰的なGreedy Decoding
- Attentionの可視化、埋め込みの初期化・スケールの比較実験

## セットアップ

Python 3.12以上と[uv](https://docs.astral.sh/uv/)を使用します。以下のコマンドはすべてリポジトリのルートで実行してください。

```bash
uv sync --locked
uv run python check_env.py
```

依存関係は`pyproject.toml`と`uv.lock`で管理しています。`uv sync`は開発用のpytest、Ruffもインストールします。環境確認スクリプトはPyTorchのバージョン、使用デバイス、勾配計算の結果を表示します。

## コピータスクを学習する

```bash
uv run python scripts/train_copy.py
```

データは実行時にランダム生成するため、外部データセットは不要です。特殊トークンは`PAD=0`、`BOS=1`、`EOS=2`で、通常のトークンには`3〜31`を使用します。

```text
Source:        [7, 15, 22, EOS, PAD]
Decoder input: [BOS, 7, 15, 22, PAD]
Labels:        [7, 15, 22, EOS, PAD]
```

標準の学習設定は次のとおりです。設定を変更する場合は`scripts/train_copy.py`を編集してください。

| 項目 | 設定 |
| --- | --- |
| 語彙数 | 32 |
| 内容トークンの系列長 | 1〜8（EOS・PADを除く） |
| モデル次元 / ヘッド数 | 32 / 4 |
| FFNの中間次元 | 128 |
| Encoder / Decoderの層数 | 各2層 |
| Dropout | 0.1 |
| バッチサイズ / 学習ステップ数 | 64 / 5,000 |
| Optimizer / 学習率 | Adam / 3e-4 |
| 損失関数 | PADを除外したCross Entropy |

500ステップごとに損失とトークン正解率を表示し、学習後は重みとモデル設定を`checkpoints/copy_transformer.pt`に保存します。同じパスのファイルは上書きされます。その後、推論例とTeacher Forcingによる検証結果を表示します。

学習・評価スクリプトはMPSが利用可能ならMPS、それ以外はCPUを選択します。`check_env.py`はCUDAも検出しますが、学習スクリプトでCUDAを使うにはデバイス選択部分の変更が必要です。

## 学習済みモデルを評価する

学習後に実行します。

```bash
uv run python scripts/evaluate_copy.py
```

- 系列長別のトークン正解率、Teacher Forcingの完全一致率、Greedy Decodingの完全一致率を表示
- 予測位置ごとの正解率をヒートマップで表示
- 最終Decoder層のCross-Attentionを`cross_attention_head_0.png`〜`cross_attention_head_3.png`に保存
- 内容トークンとEOSの正解率、埋め込みと位置エンコーディングのRMSを表示

途中でグラフのウィンドウが開きます。閉じると残りの評価が進みます。画面のない環境では次のように実行できます（位置別ヒートマップのウィンドウは表示されません）。

```bash
MPLBACKEND=Agg uv run python scripts/evaluate_copy.py
```

Teacher Forcingは正解の直前トークンを入力に使い、Greedy Decodingはモデル自身の予測を次の入力に使います。両者の完全一致率を比較すると、自己回帰推論での誤りの影響を確認できます。

## モデルの使用例

```python
import torch
from transformer.model import Transformer

model = Transformer(
    src_vocab_size=32,
    tgt_vocab_size=32,
    d_model=32,
    num_heads=4,
    d_ff=128,
    num_layers=2,
    max_len=64,
)
model.eval()

src = torch.tensor([[7, 15, 22, 2, 0]])
tgt = torch.tensor([[1, 7, 15, 22, 0]])

with torch.no_grad():
    logits = model(src, tgt)

print(logits.shape)  # torch.Size([1, 5, 32])
```

入力は`(batch_size, sequence_length)`の整数トークンID、出力は`(batch_size, target_length, tgt_vocab_size)`のlogitsです。この例は未学習モデルの入出力を確認するものです。

`d_model`は`num_heads`で割り切れる値にし、入力系列長は`max_len`以下にしてください。マスクは`True`が参照可能、`False`が遮断を表します。各入力系列には少なくとも1つの非PADトークンが必要です。`Transformer.forward`はマスクを自動生成し、Attention重みはEncoder / Decoderを直接呼ぶことで取得できます。

## 埋め込みの比較実験

```bash
# 埋め込みの初期化スケールを2条件で比較
uv run python scripts/compare_embedding_init.py

# 埋め込みと位置エンコーディングのスケールを4条件で比較
uv run python scripts/ablation_embedding.py

# アブレーション実験のCSVからグラフを生成
uv run python scripts/plot_ablation.py
```

比較実験は`results/embedding_initialization.csv`、アブレーション実験は`results/embedding_ablation.csv`と条件ごとのチェックポイントを保存します。描画スクリプトは`results/ablation_content_accuracy.png`と`results/ablation_exact_match.png`を生成します。同名の出力は上書きされます。

実験用チェックポイントはコピータスクの評価スクリプトとは保存形式が異なります。`evaluate_copy.py`で読み込む対象は`train_copy.py`が生成した`copy_transformer.pt`です。

## テスト

```bash
uv run pytest
```

Attention、PyTorchのMulti-Head Attentionとの比較、マスク、残差接続、Encoder / Decoder、モデル全体などをテストします。

## ディレクトリ構成

```text
src/transformer/
├── attention.py            # Attention / Multi-Head Attention
├── embedding.py            # トークン埋め込みと位置情報の結合
├── positional_encoding.py # 正弦波による位置エンコーディング
├── feed_forward.py         # Feed Forward Network
├── normalization.py        # Layer Normalization
├── masking.py              # Padding / Causal Mask
├── encoder.py              # Encoderブロックとスタック
├── decoder.py              # Decoderブロックとスタック
└── model.py                # Encoder–Decoderと出力射影
scripts/                    # 学習・評価・比較実験
experiments/                # Attentionの動作確認
tests/                     # 単体・結合テスト
checkpoints/               # 実行時に保存するモデル
results/                   # 実験のCSV・グラフ
```
