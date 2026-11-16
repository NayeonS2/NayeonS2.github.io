"""글 표지 이미지(1200x630)를 블로그의 '키티 핑크' 테마 색으로 만든다.

홈 화면 카드는 표지의 가운데 부분만 잘라 보여 주므로 글자는 가운데(x 180~1020) 안에 둔다.
양옆에는 attention 가중치 행렬을 닮은 분홍 격자를 장식으로 깐다.

사용법:
    .venv/bin/python tools/make_cover.py <slug> "<시리즈 라벨>" "<제목 1줄>" ["<제목 2줄>"] --sub "<작은 글씨>"
예:
    .venv/bin/python tools/make_cover.py transformer "LLM FOUNDATIONS #1" "Attention Is" "All You Need" \
        --sub "Vaswani et al., 2017  ·  The Transformer"
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG, ACCENT, INK, SUB = "#fff0f3", "#e11d48", "#3f1d26", "#9f6b78"


def heat_grid(ax, x0, y0, n_rows, n_cols, seed, size=0.42, gap=0.06):
    rng = np.random.default_rng(seed)
    w = rng.random((n_rows, n_cols)) ** 3
    w = w / w.max()
    for r in range(n_rows):
        for c in range(n_cols):
            ax.add_patch(plt.Rectangle((x0 + c * (size + gap), y0 + r * (size + gap)), size, size,
                                       color=ACCENT, alpha=0.06 + 0.5 * w[r, c], linewidth=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("label")
    ap.add_argument("title", nargs="+")
    ap.add_argument("--sub", default="")
    a = ap.parse_args()

    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    heat_grid(ax, -0.2, 0.3, 12, 3, seed=7)    # 왼쪽 장식
    heat_grid(ax, 10.75, 0.3, 12, 3, seed=11)  # 오른쪽 장식

    lines = a.title
    top = 3.55 + 0.55 * (len(lines) - 1)
    ax.text(6, top + 1.05, a.label, color=ACCENT, fontsize=19, fontweight="bold", ha="center")
    for i, line in enumerate(lines):
        ax.text(6, top - i * 1.1, line, color=INK, fontsize=46, fontweight="bold", ha="center")
    if a.sub:
        ax.text(6, top - len(lines) * 1.1 - 0.15, a.sub, color=SUB, fontsize=17, ha="center")

    out = Path(__file__).resolve().parent.parent / "assets" / "img" / "posts" / a.slug / "cover.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    print(f"saved {out.relative_to(out.parents[4])}")


if __name__ == "__main__":
    main()
