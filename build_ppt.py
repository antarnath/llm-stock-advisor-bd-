"""
Build thesis_presentation.pptx — 19 slides for the thesis defense.

Project: A Multimodal LLM-Orchestrated Financial Advisory Framework
for the Dhaka Stock Exchange.

Notes:
- No file paths / phase references in any slide.
- "Data leakage" is explained in plain English as a single slide
  titled "Why Prediction Numbers Looked Too Good — and What We Did".
- Members, supervisor, and institution are taken from the user's
  request.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

# ---------- Theme ----------
NAVY = RGBColor(0x0B, 0x2E, 0x4F)        # primary dark
TEAL = RGBColor(0x14, 0x8F, 0x77)        # accent
GOLD = RGBColor(0xE0, 0xA8, 0x16)        # highlight
LIGHT = RGBColor(0xF4, 0xF6, 0xFA)       # background tint
DARK_TEXT = RGBColor(0x1C, 0x1C, 0x1C)
GREY = RGBColor(0x55, 0x60, 0x6E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xC0, 0x39, 0x2B)


def add_textbox(slide, left, top, width, height, text, *,
                font_size=18, bold=False, color=DARK_TEXT, align=PP_ALIGN.LEFT,
                anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return tb


def add_filled_rect(slide, left, top, width, height, fill, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    return shape


def add_rounded(slide, left, top, width, height, fill, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    return shape


def add_bullets(slide, left, top, width, height, bullets, *,
                font_size=18, color=DARK_TEXT, bullet_color=TEAL, line_spacing=1.15):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        # Bullet glyph
        r1 = p.add_run()
        r1.text = "▸  "
        r1.font.size = Pt(font_size)
        r1.font.bold = True
        r1.font.color.rgb = bullet_color
        r1.font.name = "Calibri"
        # Text
        r2 = p.add_run()
        r2.text = item
        r2.font.size = Pt(font_size)
        r2.font.color.rgb = color
        r2.font.name = "Calibri"
    return tb


def add_slide_chrome(slide, slide_no, title, subtitle=None):
    """Adds the navbar-style header + slide number + footer line."""
    # Top accent bar
    add_filled_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.55), NAVY)
    # Gold thin line under navbar
    add_filled_rect(slide, Inches(0), Inches(0.55), Inches(13.333), Inches(0.06), GOLD)
    # Title
    add_textbox(slide, Inches(0.5), Inches(0.07), Inches(10.5), Inches(0.5),
                title, font_size=24, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    # Slide number pill
    pill = add_rounded(slide, Inches(12.4), Inches(0.10), Inches(0.75), Inches(0.36),
                       fill=GOLD)
    add_textbox(slide, Inches(12.4), Inches(0.10), Inches(0.75), Inches(0.36),
                str(slide_no), font_size=14, bold=True, color=NAVY,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # Subtitle (under header)
    if subtitle:
        add_textbox(slide, Inches(0.5), Inches(0.68), Inches(12.3), Inches(0.4),
                    subtitle, font_size=14, italic=True, color=GREY,
                    anchor=MSO_ANCHOR.TOP)
    # Footer
    add_filled_rect(slide, Inches(0), Inches(7.30), Inches(13.333), Inches(0.20), LIGHT)
    add_textbox(slide, Inches(0.5), Inches(7.30), Inches(12.3), Inches(0.20),
                "B.Sc. Engineering Thesis  •  Department of CSE, Faridpur Engineering College",
                font_size=9, color=GREY, align=PP_ALIGN.LEFT,
                anchor=MSO_ANCHOR.MIDDLE)


# ---------- Build ----------
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
blank = prs.slide_layouts[6]  # blank


# =============================================================================
# SLIDE 1 — TITLE
# =============================================================================
s = prs.slides.add_slide(blank)
# Background
add_filled_rect(s, 0, 0, SW, SH, NAVY)
# Decorative diagonal accent
add_filled_rect(s, 0, Inches(6.6), SW, Inches(0.9), TEAL)
add_filled_rect(s, 0, Inches(0.0), Inches(4.5), Inches(7.5), RGBColor(0x07, 0x21, 0x3A))
# Gold accent
add_filled_rect(s, Inches(0.0), Inches(2.4), Inches(0.25), Inches(2.4), GOLD)

add_textbox(s, Inches(0.8), Inches(0.6), Inches(11.5), Inches(0.5),
            "B.Sc. ENGINEERING THESIS  •  PRESENTATION",
            font_size=14, bold=True, color=GOLD)
add_textbox(s, Inches(0.8), Inches(1.0), Inches(11.5), Inches(0.5),
            "Department of Computer Science and Engineering",
            font_size=14, color=WHITE)
add_textbox(s, Inches(0.8), Inches(1.4), Inches(11.5), Inches(0.5),
            "Faridpur Engineering College, Faridpur, Bangladesh",
            font_size=14, color=WHITE)

# Main title
add_textbox(s, Inches(0.8), Inches(2.35), Inches(11.5), Inches(1.0),
            "A Multimodal LLM-Orchestrated",
            font_size=36, bold=True, color=WHITE)
add_textbox(s, Inches(0.8), Inches(2.95), Inches(11.5), Inches(1.0),
            "Financial Advisory Framework",
            font_size=36, bold=True, color=WHITE)
add_textbox(s, Inches(0.8), Inches(3.55), Inches(11.5), Inches(0.7),
            "for the Dhaka Stock Exchange",
            font_size=28, bold=True, color=GOLD)

# Tagline
add_textbox(s, Inches(0.8), Inches(4.6), Inches(11.5), Inches(0.5),
            "A leakage-controlled empirical study on daily-return forecasting,",
            font_size=16, italic=True, color=WHITE)
add_textbox(s, Inches(0.8), Inches(4.95), Inches(11.5), Inches(0.5),
            "bilingual sentiment fusion, and hosted-LLM explanation generation",
            font_size=16, italic=True, color=WHITE)

# Bottom info
add_textbox(s, Inches(0.8), Inches(6.0), Inches(11.5), Inches(0.4),
            "Submitted in partial fulfillment of the requirements for the B.Sc. Engineering Degree",
            font_size=12, color=WHITE)
add_textbox(s, Inches(0.8), Inches(6.35), Inches(11.5), Inches(0.4),
            "Session: 2026   •   Supervisor: Sameya Akter, Lecturer",
            font_size=12, italic=True, color=GOLD)


# =============================================================================
# SLIDE 2 — TEAM & SUPERVISOR
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 2, "Team & Supervisor")

# Three member cards
members = [
    ("Antar Chandra Nath", "Reg: 1001", "Roll: 1068"),
    ("Anujit Datta",       "Reg: 1026", "Roll: 1067"),
    ("Md. Sakil Ahamed Sabuj", "Reg: 738",  "Roll: 1051"),
]

card_top = Inches(1.5)
card_h = Inches(2.6)
card_w = Inches(3.85)
gap = Inches(0.35)
start_left = Inches(0.6)

for i, (name, reg, roll) in enumerate(members):
    left = start_left + (card_w + gap) * i
    card = add_rounded(s, left, card_top, card_w, card_h, fill=LIGHT, line=NAVY)
    # Top color band
    add_filled_rect(s, left, card_top, card_w, Inches(0.55), TEAL)
    add_textbox(s, left, card_top, card_w, Inches(0.55),
                f"Member {i+1}", font_size=14, bold=True, color=WHITE,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # Initial circle
    initial = name[0]
    circ = s.shapes.add_shape(MSO_SHAPE.OVAL,
                              left + card_w/2 - Inches(0.55),
                              card_top + Inches(0.75),
                              Inches(1.1), Inches(1.1))
    circ.fill.solid()
    circ.fill.fore_color.rgb = NAVY
    circ.line.color.rgb = GOLD
    circ.line.width = Pt(2.5)
    circ.shadow.inherit = False
    add_textbox(s, left + card_w/2 - Inches(0.55),
                card_top + Inches(0.75), Inches(1.1), Inches(1.1),
                initial, font_size=42, bold=True, color=WHITE,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # Name
    add_textbox(s, left + Inches(0.2), card_top + Inches(1.95),
                card_w - Inches(0.4), Inches(0.4),
                name, font_size=14, bold=True, color=NAVY,
                align=PP_ALIGN.CENTER)
    add_textbox(s, left + Inches(0.2), card_top + Inches(2.25),
                card_w - Inches(0.4), Inches(0.3),
                reg, font_size=11, color=GREY, align=PP_ALIGN.CENTER)
    add_textbox(s, left + Inches(0.2), card_top + Inches(2.50),
                card_w - Inches(0.4), Inches(0.3),
                roll, font_size=11, color=GREY, align=PP_ALIGN.CENTER)

# Supervisor bar
super_top = Inches(4.5)
add_rounded(s, Inches(0.6), super_top, Inches(12.1), Inches(1.5),
            fill=NAVY, line=GOLD)
add_textbox(s, Inches(0.9), super_top + Inches(0.15), Inches(11.5), Inches(0.4),
            "SUPERVISOR", font_size=12, bold=True, color=GOLD)
add_textbox(s, Inches(0.9), super_top + Inches(0.5), Inches(11.5), Inches(0.5),
            "Sameya Akter", font_size=22, bold=True, color=WHITE)
add_textbox(s, Inches(0.9), super_top + Inches(0.95), Inches(11.5), Inches(0.4),
            "Lecturer, Department of Computer Science and Engineering",
            font_size=14, italic=True, color=WHITE)

# Institution
add_textbox(s, Inches(0.6), Inches(6.2), Inches(12.1), Inches(0.4),
            "Faridpur Engineering College, Faridpur, Bangladesh",
            font_size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
add_textbox(s, Inches(0.6), Inches(6.55), Inches(12.1), Inches(0.4),
            "Affiliated with the University of Dhaka",
            font_size=12, italic=True, color=GREY, align=PP_ALIGN.CENTER)


# =============================================================================
# SLIDE 3 — WHY THIS PROJECT
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 3, "Why This Project?", "The real problem Bangladesh retail investors face")

# Left column — Problem
add_rounded(s, Inches(0.5), Inches(1.25), Inches(6.0), Inches(5.5), fill=LIGHT, line=NAVY)
add_filled_rect(s, Inches(0.5), Inches(1.25), Inches(6.0), Inches(0.55), NAVY)
add_textbox(s, Inches(0.5), Inches(1.25), Inches(6.0), Inches(0.55),
            "  PROBLEM",
            font_size=16, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(0.75), Inches(2.0), Inches(5.5), Inches(4.7), [
    "DSE retail investors face three structural barriers:",
    "Information asymmetry — news is priced before they see it",
    "Bilingual reporting — important stories in Bangla never reach English tools",
    "Low technical literacy — Bloomberg/TradingView assume a pro",
    "Result: most retail decisions are made on informal tips",
], font_size=15)

# Right column — Our answer
add_rounded(s, Inches(6.8), Inches(1.25), Inches(6.0), Inches(5.5), fill=LIGHT, line=TEAL)
add_filled_rect(s, Inches(6.8), Inches(1.25), Inches(6.0), Inches(0.55), TEAL)
add_textbox(s, Inches(6.8), Inches(1.25), Inches(6.0), Inches(0.55),
            "  OUR ANSWER",
            font_size=16, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(7.05), Inches(2.0), Inches(5.5), Inches(4.7), [
    "Forecast next-day return from price-only technical features",
    "Enrich the forecast with bilingual (English + Bangla) news sentiment",
    "Explain the combined signal in plain language through a hosted LLM",
    "Built for beginners, evaluated rigorously on 30 DSE stocks",
], font_size=15, bullet_color=GOLD)


# =============================================================================
# SLIDE 4 — RESEARCH QUESTION
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 4, "Research Question")

# Big quote box
add_rounded(s, Inches(0.8), Inches(1.5), Inches(11.7), Inches(2.6), fill=NAVY, line=GOLD)
add_textbox(s, Inches(0.8), Inches(1.55), Inches(11.7), Inches(0.5),
            "RESEARCH QUESTION", font_size=14, bold=True, color=GOLD,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_textbox(s, Inches(1.1), Inches(2.05), Inches(11.0), Inches(2.0),
            "Can an integrated pipeline — leakage-controlled classical ML, "
            "an LSTM deep forecaster, multimodal price-plus-sentiment fusion, "
            "and a single hosted-LLM orchestrator — produce useful "
            "one-day-ahead directional signals for 30 DSE stocks, and render "
            "those signals as actionable, beginner-friendly natural-language advice?",
            font_size=18, italic=True, color=WHITE, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE)

# Four sub-questions as cards
sub_top = Inches(4.5)
sub_h = Inches(2.2)
sub_w = Inches(2.95)
sub_gap = Inches(0.13)
sub_left = Inches(0.5)

sub_qs = [
    ("SQ1", "Classical ML Baselines",
     "What is the realistic, leakage-free performance of Linear Regression, "
     "Random Forest, XGBoost, and LightGBM on 30 DSE stocks?"),
    ("SQ2", "LSTM Deep Forecaster",
     "Does an LSTM trained on the same feature set materially outperform the "
     "best baseline on RMSE and directional accuracy?"),
    ("SQ3", "Multimodal Fusion",
     "Does fusing per-stock daily sentiment with the LSTM (early & late fusion) "
     "yield a measurable improvement?"),
    ("SQ4", "LLM Orchestrator",
     "Can a hosted LLM synthesise forecast, sentiment, and freshness into "
     "coherent, beginner-friendly recommendations?"),
]

for i, (tag, title, body) in enumerate(sub_qs):
    L = sub_left + (sub_w + sub_gap) * i
    add_rounded(s, L, sub_top, sub_w, sub_h, fill=LIGHT, line=NAVY)
    add_filled_rect(s, L, sub_top, sub_w, Inches(0.45), TEAL)
    add_textbox(s, L + Inches(0.15), sub_top, Inches(0.8), Inches(0.45),
                tag, font_size=14, bold=True, color=GOLD,
                anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, L + Inches(0.9), sub_top, sub_w - Inches(1.0), Inches(0.45),
                title, font_size=12, bold=True, color=WHITE,
                anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, L + Inches(0.15), sub_top + Inches(0.55),
                sub_w - Inches(0.3), sub_h - Inches(0.65),
                body, font_size=11, color=DARK_TEXT)


# =============================================================================
# SLIDE 5 — OBJECTIVES
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 5, "Objectives", "Four concrete deliverables, in priority order")

objs = [
    ("O1", "Leakage-Free DSE Benchmark",
     "Build a publicly documented, reproducible leakage-free benchmark on 30 DSE stocks "
     "by identifying and removing a real-world data-leakage artefact that previously "
     "inflated R² to ~0.89."),
    ("O2", "LSTM + Multimodal Models",
     "Train, ablate, and report a 3-layer LSTM and a multimodal LSTM (early + late fusion) "
     "across all 30 stocks under identical time-based splits."),
    ("O3", "Bilingual News Corpus",
     "Curate and label a bilingual English + Bangla DSE news corpus and quantify the "
     "sentiment-return correlation."),
    ("O4", "LLM Orchestrator",
     "Stand up a single hosted-LLM orchestrator that fuses the multimodal forecast with "
     "sentiment and freshness metadata into a plain-language reply."),
]

for i, (tag, head, body) in enumerate(objs):
    top = Inches(1.4 + i * 1.4)
    add_rounded(s, Inches(0.5), top, Inches(12.3), Inches(1.25), fill=LIGHT, line=NAVY)
    # tag
    tag_box = add_rounded(s, Inches(0.7), top + Inches(0.15), Inches(0.95), Inches(0.95),
                          fill=NAVY, line=GOLD)
    add_textbox(s, Inches(0.7), top + Inches(0.15), Inches(0.95), Inches(0.95),
                tag, font_size=22, bold=True, color=GOLD,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, Inches(1.85), top + Inches(0.15), Inches(10.7), Inches(0.4),
                head, font_size=18, bold=True, color=NAVY)
    add_textbox(s, Inches(1.85), top + Inches(0.55), Inches(10.7), Inches(0.7),
                body, font_size=13, color=DARK_TEXT)


# =============================================================================
# SLIDE 6 — LITERATURE IN ONE PAGE
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 6, "Literature Landscape", "Where this thesis sits in the existing research")

# Four columns
cols = [
    ("Classical & ML",
     "Linear Regression, Random Forest, XGBoost, LightGBM",
     "Robust, interpretable baselines. R² collapses near zero once leakage is removed."),
    ("Deep Learning",
     "LSTM, GRU, CNN-LSTM, Informer, Autoformer, PatchTST",
     "LSTM is the workhorse on short-horizon noisy daily returns. Transformers are "
     "expensive at this scale."),
    ("Financial LLMs",
     "FinBERT, FinGPT, VADER",
     "FinBERT leads on English sentiment (encoder-style). FinGPT is an instruction-tuned "
     "decoder alternative."),
    ("LLM Orchestration",
     "AutoGen, Generative Agents, LLM-Factor",
     "Hosted LLMs as the reasoning layer, not the forecaster — avoids numerical hallucination."),
]

# Two-row, two-col layout
col_w = Inches(6.0)
col_h = Inches(2.55)
positions = [
    (Inches(0.5), Inches(1.4)),
    (Inches(6.8), Inches(1.4)),
    (Inches(0.5), Inches(4.1)),
    (Inches(6.8), Inches(4.1)),
]
for i, (head, models, body) in enumerate(cols):
    L, T = positions[i]
    add_rounded(s, L, T, col_w, col_h, fill=LIGHT, line=NAVY)
    add_filled_rect(s, L, T, col_w, Inches(0.5), NAVY)
    add_textbox(s, L + Inches(0.2), T, col_w - Inches(0.4), Inches(0.5),
                head, font_size=16, bold=True, color=WHITE,
                anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, L + Inches(0.2), T + Inches(0.6), col_w - Inches(0.4), Inches(0.5),
                models, font_size=12, italic=True, color=TEAL)
    add_textbox(s, L + Inches(0.2), T + Inches(1.05), col_w - Inches(0.4), col_h - Inches(1.1),
                body, font_size=13, color=DARK_TEXT)

# Bottom thin bar — research gap
add_rounded(s, Inches(0.5), Inches(6.75), Inches(12.3), Inches(0.5), fill=GOLD)
add_textbox(s, Inches(0.6), Inches(6.75), Inches(12.1), Inches(0.5),
            "Research gap: leakage-controlled multi-stock DSE benchmark + bilingual "
            "sentiment + hosted-LLM orchestration — all in one open framework.",
            font_size=12, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)


# =============================================================================
# SLIDE 7 — METHODOLOGY OVERVIEW
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 7, "Methodology at a Glance",
                 "Four reproducible components, one end-to-end pipeline")

# Pipeline boxes
stages = [
    ("DATA", "30 DSE stocks\n+ 3 indices\n+ 1,560 news articles\n(2010 — 2026)", TEAL),
    ("FEATURES", "34 technical indicators\n41 multimodal features\n(price + sentiment)", NAVY),
    ("MODELS", "Linear, RF, XGBoost, LightGBM\nLSTM  (3-layer, 128 hidden)\nEarly & late fusion", GOLD),
    ("ORCHESTRATOR", "Hosted LLM\nforecast + sentiment\n+ freshness\n→ plain-language reply", TEAL),
]

stage_w = Inches(2.85)
stage_h = Inches(2.7)
gap = Inches(0.25)
start = Inches(0.7)
for i, (head, body, color) in enumerate(stages):
    L = start + (stage_w + gap) * i
    add_rounded(s, L, Inches(1.5), stage_w, stage_h, fill=LIGHT, line=color)
    add_filled_rect(s, L, Inches(1.5), stage_w, Inches(0.55), color)
    add_textbox(s, L, Inches(1.5), stage_w, Inches(0.55),
                head, font_size=16, bold=True, color=WHITE,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, L + Inches(0.15), Inches(2.15),
                stage_w - Inches(0.3), stage_h - Inches(0.7),
                body, font_size=13, color=DARK_TEXT, align=PP_ALIGN.CENTER,
                anchor=MSO_ANCHOR.MIDDLE)
    # Arrow between stages
    if i < 3:
        ax = L + stage_w + Inches(0.02)
        ay = Inches(1.5) + stage_h/2 - Inches(0.15)
        # arrow as a triangle
        tri = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, ax, ay, Inches(0.22), Inches(0.30))
        tri.fill.solid()
        tri.fill.fore_color.rgb = NAVY
        tri.line.fill.background()
        tri.shadow.inherit = False

# Bottom — what makes the pipeline honest
add_rounded(s, Inches(0.5), Inches(4.55), Inches(12.3), Inches(2.0), fill=NAVY, line=GOLD)
add_textbox(s, Inches(0.5), Inches(4.6), Inches(12.3), Inches(0.4),
            "WHAT KEEPS THE PIPELINE HONEST", font_size=13, bold=True, color=GOLD,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(0.9), Inches(5.05), Inches(11.5), Inches(1.5), [
    "Time-based 80/20 split (no shuffling, no future leakage) — applied identically to every stock.",
    "Per-stock standardiser fit on train only; frozen for inference.",
    "Lag-one features only — no access to today's close or today's news at prediction time.",
    "All numbers come from one reproducible pipeline; metrics are reported per stock, then averaged.",
], font_size=14, color=WHITE, bullet_color=GOLD)


# =============================================================================
# SLIDE 8 — THE DATA
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 8, "The Data", "What we collected, how big it is, where it comes from")

# Left — three info cards
left_x = Inches(0.5)
card_w = Inches(6.0)
card_h = Inches(1.65)
gap = Inches(0.15)

cards = [
    ("Price Data",
     "30 top-by-liquidity DSE stocks\n4,335 business days per stock\n130,050 stock-day records\n2010-01-01 → 2026-08-13"),
    ("Indices",
     "DSEX (broad)\nDS30 (top-30)\nDSES (Shariah-compliant)\nSame date range as stock data"),
    ("Bilingual News",
     "1,560 curated articles\n926 English + 634 Bangla\nSources: Daily Star, Financial Express,\nProthom Alo, Sangbad Pratidin"),
]

for i, (head, body) in enumerate(cards):
    T = Inches(1.4 + i * (card_h.emu/914400 + gap.emu/914400))
    add_rounded(s, left_x, T, card_w, card_h, fill=LIGHT, line=NAVY)
    add_filled_rect(s, left_x, T, Inches(0.20), card_h, TEAL)
    add_textbox(s, left_x + Inches(0.35), T + Inches(0.1), card_w - Inches(0.5), Inches(0.4),
                head, font_size=16, bold=True, color=NAVY)
    add_textbox(s, left_x + Inches(0.35), T + Inches(0.55), card_w - Inches(0.5), Inches(1.05),
                body, font_size=13, color=DARK_TEXT)

# Right — sector table
right_x = Inches(6.9)
add_rounded(s, right_x, Inches(1.4), Inches(5.95), Inches(5.3), fill=LIGHT, line=NAVY)
add_filled_rect(s, right_x, Inches(1.4), Inches(5.95), Inches(0.55), NAVY)
add_textbox(s, right_x, Inches(1.4), Inches(5.95), Inches(0.55),
            "SECTOR DISTRIBUTION (30 STOCKS)",
            font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE)

sectors = [
    ("Bank",       10, TEAL),
    ("Pharma",      4, NAVY),
    ("Telecom",     3, GOLD),
    ("Power",       2, TEAL),
    ("Cement",      2, NAVY),
    ("Consumer",    2, GOLD),
    ("Tobacco",     1, TEAL),
    ("Electronics", 1, NAVY),
    ("Conglomerate",1, GOLD),
    ("Fuel",        1, TEAL),
    ("Gas",         1, NAVY),
    ("Services",    1, GOLD),
]
row_top = Inches(2.05)
row_h = Inches(0.37)
for i, (name, n, color) in enumerate(sectors):
    y = row_top + row_h * i
    add_textbox(s, right_x + Inches(0.25), y, Inches(2.6), row_h,
                name, font_size=12, color=DARK_TEXT, anchor=MSO_ANCHOR.MIDDLE)
    # Bar
    bar_w = Inches(0.45) * n
    add_filled_rect(s, right_x + Inches(2.95), y + Inches(0.10),
                    bar_w, Inches(0.18), color)
    add_textbox(s, right_x + Inches(2.95) + bar_w + Inches(0.10), y,
                Inches(0.6), row_h,
                str(n), font_size=12, bold=True, color=NAVY,
                anchor=MSO_ANCHOR.MIDDLE)


# =============================================================================
# SLIDE 9 — LEAKAGE / METHODOLOGICAL HONESTY
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 9, "Why Prediction Numbers Looked Too Good — and What We Did",
                 "Methodological honesty, in plain English")

# What is "data leakage"
add_rounded(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(1.45), fill=LIGHT, line=NAVY)
add_filled_rect(s, Inches(0.5), Inches(1.4), Inches(0.25), Inches(1.45), RED)
add_textbox(s, Inches(0.9), Inches(1.5), Inches(11.7), Inches(0.4),
            "WHAT IS A DATA LEAKAGE ?",
            font_size=14, bold=True, color=RED)
add_textbox(s, Inches(0.9), Inches(1.9), Inches(11.7), Inches(0.95),
            "It is when the model is allowed to peek at information from the "
            "future during training. The model looks brilliant on paper — but it is "
            "cheating. It cannot do the same thing in real life, so the real-life "
            "performance collapses.",
            font_size=14, color=DARK_TEXT)

# Left — what happened
add_rounded(s, Inches(0.5), Inches(3.0), Inches(6.0), Inches(3.8), fill=LIGHT, line=RED)
add_filled_rect(s, Inches(0.5), Inches(3.0), Inches(6.0), Inches(0.55), RED)
add_textbox(s, Inches(0.5), Inches(3.0), Inches(6.0), Inches(0.55),
            "  WHAT WE FOUND",
            font_size=14, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(0.75), Inches(3.7), Inches(5.5), Inches(3.0), [
    "First baseline used today's close as a feature",
    "Today ≈ Tomorrow, so linear fit looked near-perfect",
    "Reported R² ≈ 0.89 — exciting but unrealistic",
    "Identified this as a temporal-leakage artefact",
    "Repeated the audit across every stage of the pipeline",
], font_size=14, color=DARK_TEXT, bullet_color=RED)

# Right — what we changed
add_rounded(s, Inches(6.8), Inches(3.0), Inches(6.0), Inches(3.8), fill=LIGHT, line=TEAL)
add_filled_rect(s, Inches(6.8), Inches(3.0), Inches(6.0), Inches(0.55), TEAL)
add_textbox(s, Inches(6.8), Inches(3.0), Inches(6.0), Inches(0.55),
            "  WHAT WE FIXED",
            font_size=14, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(7.05), Inches(3.7), Inches(5.5), Inches(3.0), [
    "Target → next-day return, not next-day price",
    "Features → lag-one only (no today's close)",
    "Time-based 80/20 split, no shuffling",
    "Scaler and checkpoints fit on train only",
    "Result: R² falls to ≈ −0.02 — the realistic value",
], font_size=14, color=DARK_TEXT, bullet_color=TEAL)


# =============================================================================
# SLIDE 10 — FEATURE ENGINEERING
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 10, "Feature Engineering", "What we feed the models")

# Two panels: price-only vs multimodal
left_x = Inches(0.5)
right_x = Inches(6.9)
panel_w = Inches(6.0)
panel_h = Inches(5.4)

# Left — Price-only
add_rounded(s, left_x, Inches(1.3), panel_w, panel_h, fill=LIGHT, line=NAVY)
add_filled_rect(s, left_x, Inches(1.3), panel_w, Inches(0.55), NAVY)
add_textbox(s, left_x, Inches(1.3), panel_w, Inches(0.55),
            "PRICE-ONLY MODEL  —  34 FEATURES",
            font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE)
feat_l = [
    "Lag-one OHLCV + daily return",
    "SMA (5, 10, 20), EMA (12, 26)",
    "MACD + signal line",
    "RSI (14), Bollinger Bands (20, 2σ)",
    "ATR (14), OBV, MFI (14)",
    "ROC (10), Williams %R (14)",
    "Stochastic K/D (14, 3), CCI (20), ADX (14)",
    "Plus price-derived ratios",
]
add_bullets(s, left_x + Inches(0.3), Inches(2.0), panel_w - Inches(0.6),
            panel_h - Inches(0.8), feat_l, font_size=14, bullet_color=NAVY)

# Right — Multimodal
add_rounded(s, right_x, Inches(1.3), panel_w, panel_h, fill=LIGHT, line=TEAL)
add_filled_rect(s, right_x, Inches(1.3), panel_w, Inches(0.55), TEAL)
add_textbox(s, right_x, Inches(1.3), panel_w, Inches(0.55),
            "MULTIMODAL MODEL  —  41 FEATURES",
            font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE)
feat_r = [
    "All 34 price features above",
    "+ 7 sentiment features",
    "   • daily FinBERT score (English)",
    "   • Bangla lexicon score (Bangla)",
    "   • sentiment momentum (3-day, 7-day)",
    "   • news volume signal",
    "   • pos / neg / neu flag",
    "Sentiment is forward-filled across the 60-day window",
]
add_bullets(s, right_x + Inches(0.3), Inches(2.0), panel_w - Inches(0.6),
            panel_h - Inches(0.8), feat_r, font_size=14, bullet_color=TEAL)

# Bottom note
add_textbox(s, Inches(0.5), Inches(6.85), Inches(12.3), Inches(0.4),
            "Every feature is computed on day t−1 only — never on day t.",
            font_size=13, italic=True, color=GREY, align=PP_ALIGN.CENTER)


# =============================================================================
# SLIDE 11 — BASELINE MODELS
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 11, "Baseline ML Models",
                 "Four classical learners, leak-free, 30 stocks, 80/20 time split")

# Table
hdr = ["Model", "RMSE", "MAE", "R²", "Dir_Acc (%)"]
rows = [
    ["Linear Regression", "0.01977", "0.01560", "−0.016", "50.0"],
    ["Random Forest",     "0.02016", "0.01592", "−0.055", "50.2"],
    ["XGBoost",           "0.02024", "0.01599", "−0.061", "50.2"],
    ["LightGBM",          "0.02051", "0.01621", "−0.088", "50.2"],
]

# Table position
tx = Inches(0.5)
ty = Inches(1.4)
col_widths = [Inches(4.0), Inches(2.0), Inches(2.0), Inches(2.0), Inches(2.3)]
row_h = Inches(0.55)

# Header row
x = tx
for i, h in enumerate(hdr):
    add_filled_rect(s, x, ty, col_widths[i], row_h, NAVY)
    add_textbox(s, x, ty, col_widths[i], row_h, h,
                font_size=14, bold=True, color=WHITE,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    x += col_widths[i]

# Data rows
for r, row in enumerate(rows):
    y = ty + row_h * (r + 1)
    x = tx
    fill = LIGHT if r % 2 == 0 else WHITE
    for i, cell in enumerate(row):
        add_filled_rect(s, x, y, col_widths[i], row_h, fill,
                        line=NAVY if i == 0 else GREY)
        is_best = (r == 0)
        add_textbox(s, x, y, col_widths[i], row_h, cell,
                    font_size=14, bold=is_best,
                    color=NAVY if is_best else DARK_TEXT,
                    align=PP_ALIGN.CENTER if i > 0 else PP_ALIGN.LEFT,
                    anchor=MSO_ANCHOR.MIDDLE)
        x += col_widths[i]

# Footnote / observations
add_rounded(s, Inches(0.5), Inches(4.85), Inches(12.3), Inches(1.95),
            fill=LIGHT, line=NAVY)
add_textbox(s, Inches(0.7), Inches(4.95), Inches(12.0), Inches(0.4),
            "KEY OBSERVATIONS", font_size=14, bold=True, color=NAVY)
add_bullets(s, Inches(0.7), Inches(5.35), Inches(12.0), Inches(1.4), [
    "Linear Regression wins on 26 out of 30 stocks — non-linear models add no measurable signal at the daily horizon.",
    "Dir_Acc for every model is statistically indistinguishable from 50% (n = 767 test days, SE ≈ ±1.8%).",
    "This is consistent with the published DSE literature — daily-return forecasting is genuinely hard.",
], font_size=13, bullet_color=TEAL)


# =============================================================================
# SLIDE 12 — DEEP LEARNING (LSTM)
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 12, "Deep Forecasting — LSTM",
                 "Three-layer LSTM, 128 hidden units, 60-day sliding window")

# Left — architecture
add_rounded(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(3.0), fill=LIGHT, line=NAVY)
add_filled_rect(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(0.5), NAVY)
add_textbox(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(0.5),
            "ARCHITECTURE", font_size=14, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(0.7), Inches(2.0), Inches(5.6), Inches(2.3), [
    "3 × LSTM layers, hidden size 128, dropout 0.2",
    "Window length 60 days, Adam optimiser (lr = 1e-3)",
    "MSE loss, early stopping (patience 20), max 100 epochs",
    "Identical hyperparameters for every stock",
], font_size=13, bullet_color=NAVY)

# Left — Results table
add_rounded(s, Inches(0.5), Inches(4.5), Inches(6.0), Inches(2.3), fill=LIGHT, line=TEAL)
add_filled_rect(s, Inches(0.5), Inches(4.5), Inches(6.0), Inches(0.5), TEAL)
add_textbox(s, Inches(0.5), Inches(4.5), Inches(6.0), Inches(0.5),
            "LSTM RESULTS (AVG OVER 30 STOCKS)",
            font_size=13, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

metrics = [
    ("Average RMSE",        "0.01977"),
    ("Average MAE",         "0.01558"),
    ("Average R²",          "−0.011"),
    ("Average Dir_Acc",     "49.8%"),
    ("Median Dir_Acc",      "50.0%"),
    ("Stocks with Dir_Acc ≥ 52%", "4 / 30"),
]
for i, (k, v) in enumerate(metrics):
    y = Inches(5.05) + Inches(0.28) * i
    add_textbox(s, Inches(0.7), y, Inches(3.6), Inches(0.3),
                k, font_size=12, color=DARK_TEXT)
    add_textbox(s, Inches(4.3), y, Inches(2.0), Inches(0.3),
                v, font_size=12, bold=True, color=NAVY, align=PP_ALIGN.RIGHT)

# Right — top / bottom stocks
add_rounded(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(2.5), fill=LIGHT, line=GOLD)
add_filled_rect(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(0.5), GOLD)
add_textbox(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(0.5),
            "TOP 3 STOCKS BY Dir_Acc",
            font_size=13, bold=True, color=NAVY,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
top3 = [
    ("BEXIMCO",  "53.7%"),
    ("BSCCL",    "52.7%"),
    ("NCCBANK",  "52.4%"),
]
for i, (n, v) in enumerate(top3):
    y = Inches(2.0) + Inches(0.55) * i
    add_textbox(s, Inches(6.95), y, Inches(3.5), Inches(0.5),
                f"{i+1}.  {n}", font_size=14, bold=True, color=NAVY)
    add_textbox(s, Inches(10.5), y, Inches(2.1), Inches(0.5),
                v, font_size=14, bold=True, color=TEAL, align=PP_ALIGN.RIGHT)

# Right — bottom 3
add_rounded(s, Inches(6.8), Inches(4.0), Inches(6.0), Inches(2.5), fill=LIGHT, line=RED)
add_filled_rect(s, Inches(6.8), Inches(4.0), Inches(6.0), Inches(0.5), RED)
add_textbox(s, Inches(6.8), Inches(4.0), Inches(6.0), Inches(0.5),
            "BOTTOM 3 STOCKS BY Dir_Acc",
            font_size=13, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
bot3 = [
    ("DSEX",    "45.2%"),
    ("RENATA",  "46.3%"),
    ("DBBL",    "46.5%"),
]
for i, (n, v) in enumerate(bot3):
    y = Inches(4.6) + Inches(0.55) * i
    add_textbox(s, Inches(6.95), y, Inches(3.5), Inches(0.5),
                f"{i+1}.  {n}", font_size=14, bold=True, color=NAVY)
    add_textbox(s, Inches(10.5), y, Inches(2.1), Inches(0.5),
                v, font_size=14, bold=True, color=RED, align=PP_ALIGN.RIGHT)


# =============================================================================
# SLIDE 13 — BILINGUAL SENTIMENT
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 13, "Bilingual Sentiment Analysis",
                 "English + Bangla news, scored by FinBERT and a Bangla lexicon")

# Left — corpus & accuracy
add_rounded(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(3.0), fill=LIGHT, line=NAVY)
add_filled_rect(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(0.5), NAVY)
add_textbox(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(0.5),
            "CORPUS & MODEL",
            font_size=14, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(0.7), Inches(2.0), Inches(5.6), Inches(2.3), [
    "1,560 curated articles (926 English + 634 Bangla)",
    "English → FinBERT pre-trained transformer",
    "Bangla → curated Bangla sentiment lexicon",
    "VADER retained as a fast fallback",
    "FinBERT accuracy vs curated labels: 91.4%",
    "Macro-F1: 0.90   •   Classes: 631 neg / 499 pos / 430 neu",
], font_size=13, bullet_color=NAVY)

# Right — sent vs return heat
add_rounded(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(3.0), fill=LIGHT, line=TEAL)
add_filled_rect(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(0.5), TEAL)
add_textbox(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(0.5),
            "SENTIMENT ↔ NEXT-DAY RETURN",
            font_size=14, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
# Mini correlation visualization
corr_lines = [
    ("Lag 0",  "−0.011",  GREY),
    ("Lag 1",  "+0.013",  GREY),
    ("Lag 3",  "+0.005",  GREY),
    ("Lag 7",  "−0.002",  GREY),
    ("Statistically significant stocks (p < 0.05)", "2 / 30",  NAVY),
]
for i, (k, v, c) in enumerate(corr_lines):
    y = Inches(2.05) + Inches(0.4) * i
    add_textbox(s, Inches(6.95), y, Inches(3.5), Inches(0.4),
                k, font_size=13, color=DARK_TEXT)
    add_textbox(s, Inches(10.45), y, Inches(2.1), Inches(0.4),
                v, font_size=13, bold=True, color=c, align=PP_ALIGN.RIGHT)

# Bottom — takeaway
add_rounded(s, Inches(0.5), Inches(4.6), Inches(12.3), Inches(2.2), fill=NAVY, line=GOLD)
add_textbox(s, Inches(0.5), Inches(4.7), Inches(12.3), Inches(0.4),
            "KEY TAKEAWAY", font_size=13, bold=True, color=GOLD,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(0.9), Inches(5.15), Inches(11.5), Inches(1.5), [
    "Sentiment works very well for what it is trained on — labelling an article's tone.",
    "But the daily sentiment signal is sparse (≈ 50 days / stock) and only weakly correlated with next-day return.",
    "Implication: sentiment's value is in EXPLAINING the forecast to the user, not in moving the forecast itself.",
], font_size=14, color=WHITE, bullet_color=GOLD)


# =============================================================================
# SLIDE 14 — MULTIMODAL FUSION
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 14, "Multimodal Fusion",
                 "Combining price with sentiment under two architectures")

# Left — Early
add_rounded(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(3.0), fill=LIGHT, line=NAVY)
add_filled_rect(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(0.5), NAVY)
add_textbox(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(0.5),
            "EARLY FUSION  (~356K params)",
            font_size=13, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(0.7), Inches(2.0), Inches(5.6), Inches(2.3), [
    "Price (27 features) + Sentiment (7 features) → concatenated at input",
    "Single 3-layer LSTM (hidden 128) consumes the merged vector",
    "More parameters — more prone to overfit on the sparse sentiment signal",
    "Trained end-to-end with the same early-stopping rule",
], font_size=13, bullet_color=NAVY)

# Right — Late
add_rounded(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(3.0), fill=LIGHT, line=TEAL)
add_filled_rect(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(0.5), TEAL)
add_textbox(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(0.5),
            "LATE FUSION  (~228K params)",
            font_size=13, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(7.0), Inches(2.0), Inches(5.6), Inches(2.3), [
    "Separate price LSTM (2 layers, hidden 128) and sentiment LSTM (1 layer, hidden 32)",
    "Their outputs are concatenated and passed through a 64-unit MLP head",
    "Fewer parameters — empirically the more stable choice on this corpus",
    "Inference is conservative: tomorrow's sentiment is frozen at last-known",
], font_size=13, bullet_color=TEAL)

# Bottom — small comparison table
add_rounded(s, Inches(0.5), Inches(4.6), Inches(12.3), Inches(2.2),
            fill=LIGHT, line=NAVY)
add_textbox(s, Inches(0.5), Inches(4.7), Inches(12.3), Inches(0.4),
            "ABLATION", font_size=13, bold=True, color=NAVY,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

ab = [("Model", "RMSE", "MAE", "R²", "Dir_Acc"),
      ("Price-only LSTM",     "0.01977", "0.01558", "−0.011", "49.8%"),
      ("Early fusion",         "0.01976", "0.01557", "−0.008", "49.7%"),
      ("Late fusion",          "0.02001", "0.01577", "−0.034", "49.8%")]
ttx = Inches(0.7)
tty = Inches(5.1)
col_w = [Inches(3.6), Inches(2.0), Inches(2.0), Inches(2.0), Inches(2.0)]
rh = Inches(0.4)
for r, row in enumerate(ab):
    is_h = (r == 0)
    xx = ttx
    for i, cell in enumerate(row):
        fill = NAVY if is_h else (LIGHT if r % 2 else WHITE)
        text_color = WHITE if is_h else (NAVY if (r == 2) else DARK_TEXT)
        bold = is_h or i == 2
        add_filled_rect(s, xx, tty + rh * r, col_w[i], rh, fill, line=GREY)
        add_textbox(s, xx, tty + rh * r, col_w[i], rh, cell,
                    font_size=12, bold=bold, color=text_color,
                    align=PP_ALIGN.CENTER if i > 0 else PP_ALIGN.LEFT,
                    anchor=MSO_ANCHOR.MIDDLE)
        xx += col_w[i]


# =============================================================================
# SLIDE 15 — HEADLINE COMPARISON
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 15, "Headline Comparison", "All components on the same 80/20 split")

# Big table
hdr = ["Component", "Model", "RMSE", "R²", "Dir_Acc"]
rows = [
    ["Baseline",   "Linear Regression",  "0.01977", "−0.016", "50.0%"],
    ["Baseline",   "Random Forest",      "0.02016", "−0.055", "50.2%"],
    ["Baseline",   "XGBoost",            "0.02024", "−0.061", "50.2%"],
    ["Baseline",   "LightGBM",           "0.02051", "−0.088", "50.2%"],
    ["Deep",       "LSTM (price only)",  "0.01977", "−0.011", "49.8%"],
    ["Multimodal", "Early fusion",       "0.01976", "−0.008", "49.7%"],
    ["Multimodal", "Late fusion",        "0.02001", "−0.034", "49.8%"],
]

tx = Inches(0.5)
ty = Inches(1.4)
col_widths = [Inches(2.5), Inches(4.0), Inches(2.0), Inches(1.8), Inches(2.0)]
row_h = Inches(0.42)

# Header
x = tx
for i, h in enumerate(hdr):
    add_filled_rect(s, x, ty, col_widths[i], row_h, NAVY)
    add_textbox(s, x, ty, col_widths[i], row_h, h,
                font_size=13, bold=True, color=WHITE,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    x += col_widths[i]

# Body
for r, row in enumerate(rows):
    y = ty + row_h * (r + 1)
    x = tx
    fill = LIGHT if r % 2 == 0 else WHITE
    for i, cell in enumerate(row):
        add_filled_rect(s, x, y, col_widths[i], row_h, fill, line=GREY)
        is_best = (r == 5)  # early fusion
        add_textbox(s, x, y, col_widths[i], row_h, cell,
                    font_size=12, bold=is_best,
                    color=NAVY if is_best else DARK_TEXT,
                    align=PP_ALIGN.CENTER if i > 1 else PP_ALIGN.LEFT,
                    anchor=MSO_ANCHOR.MIDDLE)
        x += col_widths[i]

# Takeaway
add_rounded(s, Inches(0.5), Inches(5.5), Inches(12.3), Inches(1.3), fill=NAVY, line=GOLD)
add_textbox(s, Inches(0.5), Inches(5.5), Inches(12.3), Inches(0.4),
            "HEADLINE TAKEAWAY", font_size=13, bold=True, color=GOLD,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_textbox(s, Inches(0.7), Inches(5.95), Inches(12.0), Inches(0.85),
            "All models cluster around RMSE ≈ 0.0198 and Dir_Acc ≈ 50%. The "
            "improvement from simple linear → deep multimodal is ≈ 0.05%, "
            "inside the per-stock confidence interval. The simple linear baseline "
            "is the most parsimonious adequate model on this corpus.",
            font_size=13, color=WHITE, align=PP_ALIGN.CENTER)


# =============================================================================
# SLIDE 16 — DISCUSSION
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 16, "Discussion", "Why is the market so hard to predict — and what now?")

# Three columns: Why Dir_Acc ~ 50%, Why sentiment didn't help, Threats to validity
col_w = Inches(4.0)
col_h = Inches(5.4)
gap = Inches(0.15)
top = Inches(1.4)
xs = [Inches(0.5), Inches(0.5) + col_w + gap, Inches(0.5) + 2*(col_w + gap)]
heads = [
    ("Why Dir_Acc ≈ 50%?", NAVY),
    ("Why Sentiment Didn't Help", TEAL),
    ("Threats to Validity", RED),
]
bodies = [
    [
        "Weak-form efficiency of the DSE — short-horizon returns are nearly random.",
        "On the daily horizon, news reaches the market at the same time as our feature vector.",
        "With only 767 test days, the standard error on Dir_Acc is ±1.8% — small deviations from 50% are noise.",
        "This is consistent with the published DSE literature.",
    ],
    [
        "Sentiment signal is sparse — only ~50 days per stock carry any news.",
        "Mean lag-one Pearson correlation is +0.013, statistically indistinguishable from zero.",
        "Finance papers worldwide report the same pattern for short-horizon news → return prediction.",
        "Sentiment's value is in EXPLAINING the forecast, not moving it.",
    ],
    [
        "News corpus is curated (1,560 articles) — may not represent real-time flow.",
        "Single 80/20 split — a multi-fold walk-forward validation is left to future work.",
        "Universe = 30 most-liquid stocks — results may not generalise to small-caps.",
        "Outlier clipping to ±10% removes a small tail of data-entry artefacts.",
    ],
]
for i in range(3):
    L = xs[i]
    add_rounded(s, L, top, col_w, col_h, fill=LIGHT, line=heads[i][1])
    add_filled_rect(s, L, top, col_w, Inches(0.5), heads[i][1])
    add_textbox(s, L, top, col_w, Inches(0.5),
                heads[i][0], font_size=14, bold=True, color=WHITE,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_bullets(s, L + Inches(0.2), top + Inches(0.6),
                col_w - Inches(0.4), col_h - Inches(0.7),
                bodies[i], font_size=12, bullet_color=heads[i][1])


# =============================================================================
# SLIDE 17 — LLM ORCHESTRATOR
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 17, "LLM Orchestrator — The User-Facing Layer",
                 "Where the value of the system actually lives")

# Two columns: inputs / example reply
# Left — inputs
add_rounded(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(5.4), fill=LIGHT, line=NAVY)
add_filled_rect(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(0.5), NAVY)
add_textbox(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(0.5),
            "WHAT THE LLM RECEIVES", font_size=14, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_bullets(s, Inches(0.7), Inches(2.0), Inches(5.6), Inches(4.7), [
    "Next-day return forecast (price-only LSTM)",
    "Latest sentiment score for the stock",
    "Freshness metadata — date of last news, last close",
    "Stock sector + ticker name",
    "Prompt: synthesise all of the above into a beginner-friendly reply",
    "Hosted LLM is NEVER used as the primary forecaster — only as the explainer",
], font_size=13, bullet_color=NAVY)

# Right — example reply card
add_rounded(s, Inches(6.8), Inches(1.4), Inches(6.0), Inches(5.4), fill=NAVY, line=GOLD)
add_textbox(s, Inches(6.8), Inches(1.5), Inches(6.0), Inches(0.4),
            "EXAMPLE REPLY", font_size=13, bold=True, color=GOLD,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
reply = (
    "User:  Should I buy GP today?\n\n"
    "GP forecast: +0.4% next-day return (model confidence: low)\n"
    "Latest sentiment: Positive (FinBERT 0.78, 2 articles today)\n"
    "Last data: 2026-08-13 (fresh)\n\n"
    "Advisor:\n"
    "GP is showing a slightly positive signal for tomorrow "
    "(+0.4%), supported by recent positive news. However, the "
    "forecast is weak — a single-day LSTM signal on the DSE is "
    "close to a coin flip. Do not invest based on this alone. "
    "If you already hold GP, this is not a sell signal. If you "
    "are considering a new position, size it small and consider "
    "your overall portfolio risk."
)
add_textbox(s, Inches(7.0), Inches(2.0), Inches(5.6), Inches(4.7),
            reply, font_size=12, color=WHITE)


# =============================================================================
# SLIDE 18 — CONCLUSION
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 18, "Conclusion", "What we built, what we learned, what we contribute")

# Three contribution cards
contr = [
    ("01", "Leakage-Controlled Benchmark",
     "A publicly documented, reproducible daily-return benchmark on 30 DSE "
     "stocks that future researchers can cite without inheriting the "
     "temporal-leakage artefact that inflated R² to ~0.89."),
    ("02", "Bilingual Sentiment Corpus",
     "A curated, labelled English + Bangla DSE news corpus (1,560 articles) "
     "with sentiment scores — the first reusable dataset of its kind for "
     "the Bangladesh market."),
    ("03", "LLM-Orchestrated Framework",
     "An open architecture that fuses classical ML, deep learning, "
     "multilingual sentiment, and a hosted LLM — turning a weak statistical "
     "signal into actionable, beginner-friendly explanations."),
]

card_w = Inches(4.0)
card_h = Inches(3.8)
gap = Inches(0.15)
top = Inches(1.4)
for i, (tag, head, body) in enumerate(contr):
    L = Inches(0.5) + (card_w + gap) * i
    add_rounded(s, L, top, card_w, card_h, fill=LIGHT, line=NAVY)
    add_filled_rect(s, L, top, card_w, Inches(0.6), NAVY)
    add_textbox(s, L, top, card_w, Inches(0.6),
                tag, font_size=22, bold=True, color=GOLD,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, L + Inches(0.2), top + Inches(0.75), card_w - Inches(0.4), Inches(0.5),
                head, font_size=15, bold=True, color=NAVY,
                align=PP_ALIGN.CENTER)
    add_textbox(s, L + Inches(0.2), top + Inches(1.35), card_w - Inches(0.4), card_h - Inches(1.5),
                body, font_size=13, color=DARK_TEXT, align=PP_ALIGN.LEFT)

# Bottom strap line
add_rounded(s, Inches(0.5), Inches(5.4), Inches(12.3), Inches(1.4), fill=NAVY, line=GOLD)
add_textbox(s, Inches(0.5), Inches(5.5), Inches(12.3), Inches(0.4),
            "BOTTOM LINE", font_size=13, bold=True, color=GOLD,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_textbox(s, Inches(0.7), Inches(5.9), Inches(12.0), Inches(0.85),
            "Daily-return forecasting on the DSE is genuinely hard — "
            "the simple Linear Regression baseline is competitive with the "
            "deep multimodal model. The thesis's principal contribution is the "
            "infrastructure: a leakage-free benchmark, a bilingual corpus, and "
            "an LLM orchestrator that turns weak signals into useful advice.",
            font_size=13, italic=True, color=WHITE, align=PP_ALIGN.CENTER)


# =============================================================================
# SLIDE 19 — FUTURE WORK & THANK YOU
# =============================================================================
s = prs.slides.add_slide(blank)
add_slide_chrome(s, 19, "Future Work & Thank You")

# Left — future work
add_rounded(s, Inches(0.5), Inches(1.4), Inches(7.5), Inches(5.4), fill=LIGHT, line=NAVY)
add_filled_rect(s, Inches(0.5), Inches(1.4), Inches(7.5), Inches(0.55), NAVY)
add_textbox(s, Inches(0.5), Inches(1.4), Inches(7.5), Inches(0.55),
            "FUTURE WORK", font_size=15, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
fw = [
    "Expand stock universe from 30 → full DSE (≈ 600 stocks); include DSEX as a benchmark.",
    "Add transformer-based time-series models (Informer, Autoformer, PatchTST) for longer horizons.",
    "Wire up real-time news ingestion and an LLM impact classifier for fresh signals.",
    "Add explainability (SHAP for trees, LIME for deep) so the orchestrator can quote reasons.",
    "Move from single 80/20 split to multi-fold walk-forward validation.",
    "Containerise the pipeline, deploy the chatbot, and run a sector / language bias audit.",
]
add_bullets(s, Inches(0.7), Inches(2.1), Inches(7.1), Inches(4.5),
            fw, font_size=13, bullet_color=NAVY)

# Right — thank you panel
add_rounded(s, Inches(8.2), Inches(1.4), Inches(4.6), Inches(5.4), fill=NAVY, line=GOLD)
add_textbox(s, Inches(8.2), Inches(1.8), Inches(4.6), Inches(0.6),
            "THANK YOU", font_size=30, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_textbox(s, Inches(8.2), Inches(2.5), Inches(4.6), Inches(0.5),
            "Questions are welcome.",
            font_size=14, italic=True, color=GOLD,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
# divider
add_filled_rect(s, Inches(8.9), Inches(3.2), Inches(3.2), Inches(0.04), GOLD)
add_textbox(s, Inches(8.2), Inches(3.4), Inches(4.6), Inches(0.4),
            "TEAM", font_size=12, bold=True, color=GOLD,
            align=PP_ALIGN.CENTER)
add_textbox(s, Inches(8.2), Inches(3.75), Inches(4.6), Inches(2.0),
            "Antar Chandra Nath   •   Reg 1001   •   Roll 1068\n"
            "Anujit Datta          •   Reg 1026   •   Roll 1067\n"
            "Md. Sakil Ahamed Sabuj •  Reg 738  •   Roll 1051",
            font_size=12, color=WHITE, align=PP_ALIGN.CENTER)
add_filled_rect(s, Inches(8.9), Inches(5.5), Inches(3.2), Inches(0.04), GOLD)
add_textbox(s, Inches(8.2), Inches(5.6), Inches(4.6), Inches(0.4),
            "SUPERVISOR", font_size=12, bold=True, color=GOLD,
            align=PP_ALIGN.CENTER)
add_textbox(s, Inches(8.2), Inches(5.95), Inches(4.6), Inches(0.8),
            "Sameya Akter\nLecturer, Department of CSE\nFaridpur Engineering College",
            font_size=12, color=WHITE, align=PP_ALIGN.CENTER)


# ---------- Save ----------
out = "/media/antar-chandra-nath/Media/Research/Dataset/thesis_presentation.pptx"
prs.save(out)
print(f"OK — saved {out} with {len(prs.slides)} slides")
