"""
Draw system-level architecture diagram for the LLM-Orchestrated Financial Advisor.

Shows the complete pipeline: Data → Models → Multi-Agent → User
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

C_DATA = "#0ea5e9"      # sky
C_MODEL = "#8b5cf6"     # violet
C_AGENT = "#f59e0b"     # amber
C_USER = "#10b981"      # emerald
C_OUT = "#0f172a"
BG = "#f8fafc"


def box(ax, x, y, w, h, label, fc, fontsize=9, fontcolor="white", weight="bold"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                       linewidth=1.4, edgecolor="black", facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=fontcolor)


def arrow(ax, x1, y1, x2, y2, color="black", lw=1.5, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw,
                        mutation_scale=18)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # Title
    ax.text(8, 10.4, "LLM-Orchestrated Financial Advisor — System Architecture",
            ha="center", fontsize=17, fontweight="bold", color=C_OUT)
    ax.text(8, 10.0, "End-to-end pipeline: data → models → multi-agent → transparent advice",
            ha="center", fontsize=11, color="#475569", style="italic")

    # =============================================
    # ROW 1: DATA SOURCES (bottom of pipeline)
    # =============================================
    box(ax, 0.3, 0.5, 2.0, 0.7, "DSE Prices\n(30 stocks, 2010-26)", C_DATA, fontsize=8)
    box(ax, 2.6, 0.5, 2.0, 0.7, "News Articles\n(1,560 en+bn)", C_DATA, fontsize=8)
    box(ax, 4.9, 0.5, 2.0, 0.7, "DSEX Index\n(Market context)", C_DATA, fontsize=8)

    # =============================================
    # ROW 2: FEATURE ENGINEERING
    # =============================================
    box(ax, 1.4, 1.5, 4.6, 0.8, "Feature Engineering (Phase 2)\n27 technical indicators + 7 sentiment features", C_DATA, fontsize=9)

    # =============================================
    # ROW 3: PREDICTION MODELS
    # =============================================
    box(ax, 0.3, 2.7, 2.0, 1.1, "Baseline ML\n(Phase 3)\nLR / XGBoost", C_MODEL, fontsize=8)
    box(ax, 2.5, 2.7, 2.0, 1.1, "Deep Learning\n(Phase 4-5)\nLSTM / Transformer", C_MODEL, fontsize=8)
    box(ax, 4.7, 2.7, 2.0, 1.1, "Multimodal LSTM\n(Phase 7)\nEarly/Late/Attention", C_MODEL, fontsize=8)

    # =============================================
    # ROW 4: XAI LAYER
    # =============================================
    box(ax, 1.4, 4.1, 4.6, 0.8, "Explainable AI (Phase 8) — SHAP / LIME", "#ef4444", fontsize=10)

    # =============================================
    # ROW 5: MULTI-AGENT SYSTEM (signature contribution)
    # =============================================
    ax.text(0.3, 6.2, "Multi-Agent System (Phase 10)", fontsize=11, fontweight="bold", color=C_OUT)
    ax.text(0.3, 5.95, "Each agent = specialist. Orchestrator = debate.", fontsize=8, color="#475569", style="italic")

    # 4 agents
    box(ax, 0.3, 5.0, 1.6, 0.9, "Technical\nAnalyst\nAgent", C_AGENT, fontsize=8)
    box(ax, 2.0, 5.0, 1.6, 0.9, "News\nAnalyst\nAgent", C_AGENT, fontsize=8)
    box(ax, 3.7, 5.0, 1.6, 0.9, "Risk\nAnalyst\nAgent", C_AGENT, fontsize=8)
    box(ax, 5.4, 5.0, 1.6, 0.9, "Portfolio\nManager\nAgent", C_AGENT, fontsize=8)

    # Orchestrator
    box(ax, 7.2, 5.0, 1.8, 0.9, "Orchestrator\n(LLM debate)", "#a855f7", fontsize=9)

    # RAG agent (Phase 9)
    box(ax, 9.2, 5.0, 1.6, 0.9, "RAG Agent\n(News retrieval)", C_AGENT, fontsize=8)

    # =============================================
    # ROW 6: OUTPUTS
    # =============================================
    box(ax, 11.2, 5.0, 1.8, 0.9, "Final\nRecommendation\n+ Reasoning", C_USER, fontsize=8)

    # =============================================
    # ROW 7: USER INTERFACES
    # =============================================
    box(ax, 1.4, 7.0, 4.6, 0.8, "REST API (Phase 12)\nFastAPI + JWT", C_USER, fontsize=9)
    box(ax, 7.5, 7.0, 4.6, 0.8, "Web Dashboard (Phase 13)\nStreamlit / Next.js", C_USER, fontsize=9)
    box(ax, 13.0, 7.0, 2.6, 0.8, "Daily Telegram\nBrief", C_USER, fontsize=9)

    # =============================================
    # ROW 8: END USER
    # =============================================
    box(ax, 5.5, 8.4, 4.5, 1.0, "Retail Investor\n(Bangladeshi, ৳50K - ৳50L portfolio)",
        "#1e293b", fontsize=10)

    # =============================================
    # ARROWS
    # =============================================
    # Data → FE
    arrow(ax, 1.3, 1.2, 2.0, 1.5)
    arrow(ax, 3.6, 1.2, 3.8, 1.5)
    arrow(ax, 5.9, 1.2, 5.4, 1.5)

    # FE → Models
    arrow(ax, 2.5, 2.3, 1.3, 2.7)
    arrow(ax, 3.7, 2.3, 3.5, 2.7)
    arrow(ax, 4.7, 2.3, 5.7, 2.7)

    # Models → XAI
    arrow(ax, 1.3, 3.8, 2.5, 4.1)
    arrow(ax, 3.5, 3.8, 3.7, 4.1)
    arrow(ax, 5.7, 3.8, 5.0, 4.1)

    # XAI → Agents
    arrow(ax, 3.7, 4.9, 1.1, 5.0)
    arrow(ax, 3.7, 4.9, 2.8, 5.0)
    arrow(ax, 3.7, 4.9, 4.5, 5.0)
    arrow(ax, 3.7, 4.9, 6.2, 5.0)

    # Agents → Orchestrator
    arrow(ax, 1.9, 5.45, 7.2, 5.45)
    arrow(ax, 3.6, 5.45, 7.2, 5.45)
    arrow(ax, 5.3, 5.45, 7.2, 5.45)
    arrow(ax, 7.0, 5.45, 7.2, 5.45)

    # RAG → Orchestrator (bidirectional)
    arrow(ax, 9.2, 5.45, 9.0, 5.45)

    # Orchestrator → output
    arrow(ax, 9.0, 5.45, 11.2, 5.45)

    # Output → Interfaces
    arrow(ax, 12.0, 5.9, 3.7, 7.0, color=C_USER, lw=2)
    arrow(ax, 12.1, 5.45, 9.8, 7.0, color=C_USER, lw=2)
    arrow(ax, 12.2, 5.0, 14.3, 7.0, color=C_USER, lw=2)

    # Interfaces → User
    arrow(ax, 3.7, 7.8, 6.5, 8.4, color=C_OUT, lw=2)
    arrow(ax, 9.8, 7.8, 8.5, 8.4, color=C_OUT, lw=2)
    arrow(ax, 14.3, 7.8, 9.5, 8.4, color=C_OUT, lw=2)

    # Feedback loop
    arrow(ax, 6.0, 8.4, 1.7, 1.0, color="#94a3b8", lw=1, style="-|>")
    ax.text(2.8, 4.5, "feedback", fontsize=7, color="#94a3b8", rotation=35)

    # Legend
    legend = [
        (C_DATA, "Data & Features"),
        (C_MODEL, "Prediction Models"),
        ("#ef4444", "Explainability"),
        (C_AGENT, "Multi-Agent Specialists"),
        ("#a855f7", "Orchestrator"),
        (C_USER, "User-Facing Layer"),
    ]
    for i, (color, label) in enumerate(legend):
        x = 0.3 + i * 2.6
        ax.add_patch(FancyBboxPatch((x, 9.3), 0.3, 0.3, boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor="black", lw=1))
        ax.text(x + 0.4, 9.45, label, fontsize=8, va="center", color=C_OUT)

    out_png = "docs/figures/system_architecture.png"
    out_pdf = "docs/figures/system_architecture.pdf"
    os.makedirs("docs/figures", exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=BG)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=BG)
    print(f"💾 Saved {out_png}")
    print(f"💾 Saved {out_pdf}")


if __name__ == "__main__":
    main()
