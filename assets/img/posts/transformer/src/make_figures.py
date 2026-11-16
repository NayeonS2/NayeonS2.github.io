"""Transformer 리뷰 글의 차트와 표지를 만든다.

실행: .venv/bin/python assets/img/posts/transformer/src/make_figures.py
수치 출처: Vaswani et al., 2017 (arXiv:1706.03762v7) Table 2, Table 3, Section 3.5, Section 5.3
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, GREEN, PURPLE, GRAY, INK = "#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#6b7280", "#1f2937"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    print("saved", name)


# 1. EN-DE BLEU vs training cost (Table 2, single models with both values reported)
models = [
    ("GNMT + RL", 24.6, 2.3e19, GRAY),
    ("ConvS2S", 25.16, 9.6e18, GRAY),
    ("MoE", 26.03, 2.0e19, GRAY),
    ("GNMT + RL Ens.", 26.30, 1.8e20, GRAY),
    ("ConvS2S Ens.", 26.36, 7.7e19, GRAY),
    ("Transformer (base)", 27.3, 3.3e18, BLUE),
    ("Transformer (big)", 28.4, 2.3e19, ORANGE),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for name, bleu, flops, color in models:
    ax.scatter(flops, bleu, s=90 if color != GRAY else 55, color=color, zorder=3,
               edgecolor="white", linewidth=1.2)
    dx, dy, ha = 1.18, 0.0, "left"
    if name == "ConvS2S Ens.":
        dx, dy, ha = 1.0, 0.28, "center"
    if name == "GNMT + RL Ens.":
        dx, dy, ha = 1.0, -0.28, "center"
    ax.annotate(name, (flops * dx, bleu + dy), va="center", ha=ha, fontsize=9.5,
                color=INK if color == GRAY else color, fontweight="bold" if color != GRAY else "normal")
ax.set_xscale("log")
ax.set_xlim(1.5e18, 6e20)
ax.set_ylim(24, 29)
ax.set_xlabel("Training cost (FLOPs, log scale)")
ax.set_ylabel("BLEU (WMT14 EN-DE)")
ax.grid(True, which="major", color="#e5e7eb", linewidth=0.8)
ax.set_title("Higher BLEU at lower training cost", color=INK, loc="left", fontsize=12)
save(fig, "bleu_vs_cost.png")


# 2. Number of heads ablation (Table 3 rows base + (A), computation kept constant)
heads = [1, 4, 8, 16, 32]
bleu = [24.9, 25.5, 25.8, 25.8, 25.4]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(heads, bleu, color=GREEN, marker="o", linewidth=2, markersize=7, zorder=3)
for h, b in zip(heads, bleu):
    ax.annotate(f"{b}", (h, b), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5, color=INK)
ax.set_xscale("log", base=2)
ax.set_xticks(heads)
ax.set_xticklabels([f"h={h}\n(d_k={512 // h})" for h in heads])
ax.set_ylim(24.6, 26.2)
ax.set_ylabel("BLEU (newstest2013, dev)")
ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
ax.set_title("Too few or too many heads both hurt", color=INK, loc="left", fontsize=12)
save(fig, "heads_ablation.png")


# 3. Sinusoidal positional encoding (Section 3.5)
d_model, max_len = 128, 64
pos = np.arange(max_len)[:, None]
i = np.arange(0, d_model, 2)[None, :]
angle = pos / np.power(10000, i / d_model)
pe = np.zeros((max_len, d_model))
pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
im = ax.imshow(pe, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
ax.set_xlabel("Embedding dimension")
ax.set_ylabel("Position")
ax.set_title("Low dims change fast, high dims change slowly", color=INK, loc="left", fontsize=12)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.outline.set_edgecolor(GRAY)
save(fig, "positional_encoding.png")


# 4. Learning-rate schedule (Section 5.3): d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
steps = np.arange(1, 100_001)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for warmup, color in ((4000, BLUE), (8000, PURPLE)):
    lr = 512 ** -0.5 * np.minimum(steps ** -0.5, steps * warmup ** -1.5)
    ax.plot(steps, lr * 1e3, color=color, linewidth=2, label=f"warmup_steps={warmup}")
peak = 512 ** -0.5 * 4000 ** -0.5
ax.annotate(f"peak {peak:.2e} @ step 4000", (4000, peak * 1e3), textcoords="offset points",
            xytext=(18, -4), fontsize=9.5, color=BLUE)
ax.set_xlabel("Training step")
ax.set_ylabel("Learning rate (x 1e-3)")
ax.legend(frameon=False)
ax.grid(True, color="#e5e7eb", linewidth=0.8)
ax.set_title("Linear warmup, then inverse square-root decay", color=INK, loc="left", fontsize=12)
save(fig, "lr_schedule.png")


# 5. Cover: tools/make_cover.py 로 만든다 (블로그 테마 색과 맞추기 위해 분리).

# sanity: peak LR used in the post text
assert math.isclose(512 ** -0.5 * 4000 ** -0.5, 7.0e-4, rel_tol=0.02)
