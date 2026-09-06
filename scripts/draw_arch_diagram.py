"""
Draw publication-quality architecture diagram for MultimodalLSTMAttention.

Generates:
  docs/figures/multimodal_attention_architecture.png
  docs/figures/multimodal_attention_architecture.pdf

The diagram shows:
- Input: Price sequence (60 × 27) + Sentiment sequence (60 × 7)
- Price LSTM (27 → 128, 2 layers) → hidden states h_layer0, h_layer1
- Sentiment LSTM (7 → 32, 1 layer) → hidden state h_sent[-1]
- Gated cross-modal attention: q_proj (32 → 128) → sigmoid gate
- Fusion: concat [h_layer0, h_layer1 ⊙ gate]
- MLP head: 256 → 64 → 1
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

# Colors
C_PRICE = "#3b82f6"        # blue
C_SENT = "#f97316"         # orange
C_FUSE = "#22c55e"         # green
C_GATE = "#a855f7"         # purple
C_HEAD = "#ef4444"         # red
C_IN = "#1e293b"           # dark slate
C_OUT = "#0f172a"
BG = "#f8fafc"


def box(ax, x, y, w, h, label, fc, fontsize=10, fontcolor="white", weight="bold"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       linewidth=1.5, edgecolor="black", facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=fontcolor)


def arrow(ax, x1, y1, x2, y2, color="black", lw=1.5, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw,
                        mutation_scale=15)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # Title
    ax.text(7, 9.5, "MultimodalLSTMAttention — Architecture",
            ha="center", fontsize=16, fontweight="bold", color=C_IN)
    ax.text(7, 9.1, "Gated cross-modal attention for price + sentiment fusion",
            ha="center", fontsize=11, color="#475569", style="italic")

    # Inputs (left column)
    box(ax, 0.3, 7.0, 2.4, 0.9, "Price Sequence\n(60 days × 27 features)", C_PRICE, fontsize=9)
    box(ax, 0.3, 4.5, 2.4, 0.9, "Sentiment Sequence\n(60 days × 7 features)", C_SENT, fontsize=9)

    # LSTMs (middle-left)
    box(ax, 3.5, 7.0, 2.4, 1.0, "Price LSTM\n27 → 128, 2 layers", C_PRICE, fontsize=9)
    box(ax, 3.5, 4.5, 2.4, 1.0, "Sentiment LSTM\n7 → 32, 1 layer", C_SENT, fontsize=9)

    # Hidden states
    box(ax, 6.5, 7.6, 2.0, 0.6, "h_layer0, h_layer1\n(128-d each)", C_PRICE, fontsize=8)
    box(ax, 6.5, 4.6, 2.0, 0.6, "h_sent[-1]\n(32-d)", C_SENT, fontsize=8)

    # Cross-modal attention block (right column, upper)
    box(ax, 9.2, 6.0, 2.6, 1.4,
        "Cross-modal gate\nq = q_proj(h_sent)  [32→128]\nσ = sigmoid(q)\ngate ⊙ h_layer1",
        C_GATE, fontsize=8)

    # Fusion (concat)
    box(ax, 6.5, 5.5, 2.0, 0.6, "Concat\n[ h_layer0 ‖ h_layer1 ⊙ gate ]", C_FUSE, fontsize=8)

    # MLP Head
    box(ax, 9.2, 4.0, 2.6, 1.4,
        "MLP Head\nLinear 256 → 64 + ReLU + Dropout\nLinear 64 → 1",
        C_HEAD, fontsize=8)

    # Output
    box(ax, 12.4, 4.3, 1.3, 0.7, "Predicted\nReturn", "#0f172a", fontsize=10)

    # Arrows
    # Input → LSTM
    arrow(ax, 2.7, 7.45, 3.5, 7.45, color=C_PRICE, lw=2)
    arrow(ax, 2.7, 4.95, 3.5, 4.95, color=C_SENT, lw=2)

    # LSTM → hidden states
    arrow(ax, 5.9, 7.5, 6.5, 7.9, color=C_PRICE, lw=2)
    arrow(ax, 5.9, 5.0, 6.5, 4.9, color=C_SENT, lw=2)

    # h_sent → q_proj → gate
    arrow(ax, 8.5, 4.9, 9.2, 6.5, color=C_GATE, lw=2)

    # h_layer1 → gate (gets multiplied)
    arrow(ax, 8.5, 7.7, 9.2, 7.0, color=C_PRICE, lw=2)

    # gated h_layer1 + h_layer0 → concat
    arrow(ax, 9.2, 6.0, 8.5, 5.8, color=C_GATE, lw=2)
    arrow(ax, 7.5, 7.6, 7.5, 6.1, color=C_PRICE, lw=1.5)

    # concat → head
    arrow(ax, 8.5, 5.5, 9.2, 4.7, color=C_FUSE, lw=2)

    # head → output
    arrow(ax, 11.8, 4.7, 12.4, 4.65, color=C_HEAD, lw=2)

    # Caption / param count
    ax.text(7, 2.5, "Total parameters: 238,465  •  Price stream: ~166K  •  Sent stream: ~5K  •  Gate+Head: ~67K",
            ha="center", fontsize=10, color="#475569")
    ax.text(7, 1.9, "Fusion: price is gated by sentiment-derived query — sentiment modulates which",
            ha="center", fontsize=9, color="#475569")
    ax.text(7, 1.5, "price-pattern features survive into the prediction head.",
            ha="center", fontsize=9, color="#475569")

    # Legend
    legend_items = [
        (C_PRICE, "Price stream"),
        (C_SENT, "Sentiment stream"),
        (C_GATE, "Cross-modal gate"),
        (C_FUSE, "Fusion"),
        (C_HEAD, "Prediction head"),
    ]
    for i, (color, label) in enumerate(legend_items):
        x = 1 + i * 2.5
        box(ax, x, 0.5, 0.3, 0.3, "", color, fontsize=1)
        ax.text(x + 0.4, 0.65, label, fontsize=9, va="center", color=C_IN)

    out_png = "docs/figures/multimodal_attention_architecture.png"
    out_pdf = "docs/figures/multimodal_attention_architecture.pdf"
    import os
    os.makedirs("docs/figures", exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=BG)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=BG)
    print(f"💾 Saved {out_png}")
    print(f"💾 Saved {out_pdf}")


if __name__ == "__main__":
    main()
