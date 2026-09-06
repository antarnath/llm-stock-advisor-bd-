"""
Draw the multi-agent flow diagram — matches the user's reference layout:
USER → Chatbot → Orchestrator → 3 specialist agents (Stock / News / RAG)
→ each uses its own tool (LSTM / Sentiment / RAG Documents) → all converge
to LLM Reasoning → Final Explanation.

Output: docs/figures/multi_agent_flow.png + .pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Color palette
C_USER = "#0f172a"        # slate-900
C_CHATBOT = "#1e293b"     # slate-800
C_ORCH = "#a855f7"        # purple-500
C_AGENT = "#f59e0b"       # amber-500
C_TOOL = "#0ea5e9"        # sky-500
C_REASON = "#ef4444"      # red-500
C_OUT = "#10b981"         # emerald-500
C_ARROW = "#475569"       # slate-600
BG = "#f8fafc"
TXT = "#0f172a"


def box(ax, x, y, w, h, label, fc, fontsize=10, fontcolor="white", weight="bold"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1",
                       linewidth=1.4, edgecolor="black", facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=fontcolor)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.6, style="-|>"):
    color = color or C_ARROW
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw,
                        mutation_scale=18)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11, 13))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # Title
    ax.text(5.5, 12.5, "Multi-Agent Financial Advisory — Flow",
            ha="center", fontsize=17, fontweight="bold", color=TXT)
    ax.text(5.5, 12.1, "How a user query becomes a grounded recommendation",
            ha="center", fontsize=10, color="#475569", style="italic")

    # ===========================================================
    # Layout (matching reference diagram):
    #   3-column structure with the orchestrator on top branching
    #   into 3 specialist agents, each with their own tool, then
    #   converging into LLM Reasoning → Final Explanation.
    # ===========================================================

    # Column x-centers: 1.85, 5.5, 9.15   (3 equal columns)

    # ---------------------------------------------------------------
    # LAYER 1: USER
    # ---------------------------------------------------------------
    box(ax, 4.0, 10.85, 3.0, 0.8, "USER", C_USER, fontsize=14)
    arrow(ax, 5.5, 10.85, 5.5, 10.15)

    # ---------------------------------------------------------------
    # LAYER 2: CHATBOT
    # ---------------------------------------------------------------
    box(ax, 4.0, 9.35, 3.0, 0.8, "Financial Chatbot", C_CHATBOT, fontsize=11)
    arrow(ax, 5.5, 9.35, 5.5, 8.65)

    # ---------------------------------------------------------------
    # LAYER 3: LLM ORCHESTRATOR
    # ---------------------------------------------------------------
    box(ax, 4.0, 7.85, 3.0, 0.8, "LLM Orchestrator", C_ORCH, fontsize=12)
    # 3-way branch
    arrow(ax, 4.7, 7.85, 1.85, 6.85)
    arrow(ax, 5.5, 7.85, 5.5, 6.85)
    arrow(ax, 6.3, 7.85, 9.15, 6.85)

    # ---------------------------------------------------------------
    # LAYER 4: 3 SPECIALIST AGENTS
    # ---------------------------------------------------------------
    box(ax, 0.35, 5.85, 3.0, 1.0, "Stock\nAgent", C_AGENT, fontsize=12)
    box(ax, 4.0, 5.85, 3.0, 1.0, "News\nAgent", C_AGENT, fontsize=12)
    box(ax, 7.65, 5.85, 3.0, 1.0, "RAG\nAgent", C_AGENT, fontsize=12)

    # agent → tool arrows
    arrow(ax, 1.85, 5.85, 1.85, 5.15)
    arrow(ax, 5.5, 5.85, 5.5, 5.15)
    arrow(ax, 9.15, 5.85, 9.15, 5.15)

    # ---------------------------------------------------------------
    # LAYER 5: 3 TOOLS (one per agent)
    # ---------------------------------------------------------------
    box(ax, 0.35, 4.15, 3.0, 1.0, "LSTM\nForecast", C_TOOL, fontsize=11)
    box(ax, 4.0, 4.15, 3.0, 1.0, "Sentiment\nAnalysis", C_TOOL, fontsize=11)
    box(ax, 7.65, 4.15, 3.0, 1.0, "Financial\nDocuments", C_TOOL, fontsize=11)

    # convergence: tools merge back into the orchestrator column → LLM Reasoning
    # 3 arrows converging into one at x=5.5, y=3.15
    arrow(ax, 1.85, 4.15, 4.7, 3.15)
    arrow(ax, 5.5, 4.15, 5.5, 3.15)
    arrow(ax, 9.15, 4.15, 6.3, 3.15)

    # ---------------------------------------------------------------
    # LAYER 6: LLM REASONING
    # ---------------------------------------------------------------
    box(ax, 4.0, 2.35, 3.0, 0.8, "LLM Reasoning", C_REASON, fontsize=12)
    arrow(ax, 5.5, 2.35, 5.5, 1.65)

    # ---------------------------------------------------------------
    # LAYER 7: FINAL EXPLANATION
    # ---------------------------------------------------------------
    box(ax, 4.0, 0.85, 3.0, 0.8, "Final Explanation", C_OUT, fontsize=12)

    # ---------------------------------------------------------------
    # Side annotations — concrete tools the user has actually built
    # ---------------------------------------------------------------
    # Sub-caption under each agent box (small italic text)
    captions = [
        (1.85, 5.65, "load per-stock\nLSTM/XGBoost model"),
        (5.5,  5.65, "FinBERT + BanglaBERT\non 1,560 news articles"),
        (9.15, 5.65, "FAISS over\nmultilingual news"),
    ]
    for x, y, t in captions:
        ax.text(x, y, t, ha="center", va="top", fontsize=7,
                color="#475569", style="italic")

    # ---------------------------------------------------------------
    # Legend (top-right corner)
    # ---------------------------------------------------------------
    legend = [
        (C_USER,    "User"),
        (C_CHATBOT, "Chatbot UI"),
        (C_ORCH,    "Orchestrator"),
        (C_AGENT,   "Specialist Agent"),
        (C_TOOL,    "Tool / Model"),
        (C_REASON,  "Reasoning"),
        (C_OUT,     "Final Output"),
    ]
    for i, (color, label) in enumerate(legend):
        x = 8.5
        y = 12.6 - i * 0.32
        ax.add_patch(FancyBboxPatch((x, y), 0.28, 0.22,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor="black", lw=1))
        ax.text(x + 0.4, y + 0.11, label, fontsize=8, va="center", color=TXT)

    # Save
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "multi_agent_flow.png"
    out_pdf = out_dir / "multi_agent_flow.pdf"
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=BG)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()