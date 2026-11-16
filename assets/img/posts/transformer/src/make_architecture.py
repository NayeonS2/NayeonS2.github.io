"""Transformer 데이터 흐름 그림(architecture.svg)을 만든다.

본문 폭(약 700~800px)에서 글자가 작아지지 않도록 가로 860 단위로 좁게 배치한다.
각 sublayer 상자 아래쪽 회색 띠가 잔차 연결 + 층 정규화(Add & Norm)를 나타낸다.
사용법: python make_architecture.py  ->  ../architecture.svg
"""

from pathlib import Path

BLUE, ORANGE, GREEN, PURPLE, GRAY, INK = "#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#6b7280", "#1f2937"
FILL = {BLUE: "#eff6ff", ORANGE: "#fff7ed", GREEN: "#dcfce7", PURPLE: "#ede9fe"}
out = []


def text(x, y, s, size=15, color=INK, anchor="middle", bold=False):
    w = ' font-weight="bold"' if bold else ""
    s = s.replace("&", "&amp;")
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}"{w}>{s}</text>')


def box(x, y, w, h, color, lines, sub=None):
    """입출력 상자: 굵은 줄(lines) + 회색 보조 줄(sub)."""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{FILL[color]}" stroke="{color}" stroke-width="1.6"/>')
    rows = [(s, 15, INK) for s in lines] + [(s, 12.5, GRAY) for s in (sub or [])]
    y0 = y + h / 2 - (len(rows) - 1) * 9 + 5
    for i, (s, size, c) in enumerate(rows):
        text(x + w / 2, y0 + i * 18, s, size, c)


def sublayer(x, y, w, color, lines, thick=False):
    """sublayer 상자(높이 66). 아래 20px 회색 띠가 Add & Norm."""
    h, r, band = 66, 8, 20
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{FILL[color]}"/>')
    yb = y + h - band
    out.append(f'<path d="M{x},{yb} H{x + w} V{y + h - r} Q{x + w},{y + h} {x + w - r},{y + h} '
               f'H{x + r} Q{x},{y + h} {x},{y + h - r} Z" fill="#e5e7eb"/>')
    sw = 2.6 if thick else 1.6
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="none" stroke="{color}" stroke-width="{sw}"/>')
    out.append(f'<line x1="{x}" y1="{yb}" x2="{x + w}" y2="{yb}" stroke="{color}" stroke-width="1"/>')
    y0 = y + (h - band) / 2 - (len(lines) - 1) * 8.5 + 5
    for i, s in enumerate(lines):
        text(x + w / 2, y0 + i * 17, s, 14.5)
    text(x + w / 2, yb + 14.5, "Add & Norm", 11.5, "#4b5563")


def arrow(x1, y1, x2, y2, color=GRAY, marker="arr"):
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#{marker})"/>')


def layer_frame(x, y, w, h, color):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#f8fafc" stroke="{color}" stroke-width="1.6" stroke-dasharray="7 5"/>')
    text(x + w - 12, y + 19, "× N = 6 layers", 13.5, color, "end", bold=True)


W, H = 860, 490
out.append(f'<rect x="3" y="3" width="{W - 6}" height="{H - 6}" rx="18" fill="#ffffff" stroke="#e5e7eb" stroke-width="2"/>')

# ---------------- encoder (row center y=112) ----------------
text(24, 40, "ENCODER", 17, BLUE, "start", bold=True)
text(122, 40, "reads the whole source sentence at once", 14, GRAY, "start")
box(24, 74, 150, 76, BLUE, ["Source tokens"], ["embedding", "+ positional enc."])
layer_frame(204, 58, 352, 112, BLUE)
sublayer(222, 88, 150, GREEN, ["Multi-Head", "Self-Attention"])
sublayer(406, 88, 132, PURPLE, ["Feed-Forward"])
box(592, 74, 118, 76, BLUE, ["Encoder", "output z"])
arrow(174, 112, 220, 112)
arrow(372, 110, 404, 110)
arrow(538, 110, 590, 110)

# ---------------- decoder (row center y=334) ----------------
text(24, 252, "DECODER", 17, ORANGE, "start", bold=True)
text(122, 252, "writes the target sentence one token at a time", 14, GRAY, "start")
box(24, 290, 150, 90, ORANGE, ["Target tokens"], ["so far (shifted right)", "embedding + pos. enc."])
layer_frame(204, 280, 520, 112, ORANGE)
sublayer(222, 310, 150, GREEN, ["Masked", "Self-Attention"])
sublayer(406, 310, 130, GREEN, ["Cross-", "Attention"], thick=True)
sublayer(570, 310, 132, PURPLE, ["Feed-Forward"])
box(744, 296, 96, 78, ORANGE, ["Linear", "+ Softmax"], ["next token"])
arrow(174, 335, 220, 335)
arrow(372, 332, 404, 332)
arrow(536, 332, 568, 332)
arrow(702, 332, 742, 332)

# encoder output z -> cross-attention (drawn after the boxes so it stays on top)
out.append(f'<path d="M651,150 L651,214 L471,214 L471,308" fill="none" stroke="{GREEN}" stroke-width="2.6" marker-end="url(#arrG)"/>')
text(561, 205, "keys & values from z", 14, GREEN, bold=True)

# autoregressive loop at inference
out.append(f'<path d="M792,374 L792,436 L99,436 L99,382" fill="none" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 5" marker-end="url(#arrO)"/>')
text(445, 464, "at inference: the predicted token is appended and the decoder runs again", 13.5, ORANGE)

defs = "".join(
    f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" '
    f'markerWidth="{sz}" markerHeight="{sz}" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    for mid, c, sz in (("arr", GRAY, 10), ("arrG", GREEN, 12), ("arrO", ORANGE, 10))
)
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="\'Helvetica Neue\', Helvetica, Arial, sans-serif">\n<defs>{defs}</defs>\n'
       + "\n".join(out) + "\n</svg>\n")
Path(__file__).resolve().parent.parent.joinpath("architecture.svg").write_text(svg)
print("wrote architecture.svg")
