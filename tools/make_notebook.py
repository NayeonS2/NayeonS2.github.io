"""code/<slug>/demo.py를 Colab에서 바로 실행할 수 있는 노트북(demo.ipynb)으로 바꾼다.

- "# ---- " 로 시작하는 줄을 기준으로 셀을 나눈다. 그 앞부분(import, 함수 정의)은 첫 코드 셀이 된다.
- 맨 위에 글 링크와 실행 방법을 적은 설명 셀을 넣는다.
- 만든 뒤 셀을 순서대로 실행해서 출력이 code/<slug>/output.txt와 같은지 검사한다.

사용법:
    .venv/bin/python tools/make_notebook.py <slug> "<글 제목>" <글 주소 경로>
예:
    .venv/bin/python tools/make_notebook.py transformer "Attention Is All You Need 논문 리뷰" /posts/transformer/
"""

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://nayeons2.github.io"


def split_cells(source: str) -> list[str]:
    cells, cur = [], []
    for line in source.splitlines():
        if line.startswith("# ---- ") and cur:
            cells.append("\n".join(cur).strip("\n"))
            cur = []
        cur.append(line)
    cells.append("\n".join(cur).strip("\n"))
    return [c for c in cells if c.strip()]


def to_ipynb_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main():
    slug, title, path = sys.argv[1], sys.argv[2], sys.argv[3]
    demo = ROOT / "code" / slug / "demo.py"
    code_cells = split_cells(demo.read_text())

    intro = (
        f"# {title}: 직접 해보기\n\n"
        f"[블로그 글]({SITE}{path})의 \"직접 해보기\" 코드를 그대로 옮긴 노트북이다.\n\n"
        "위에서부터 셀을 차례로 실행하면 된다(`Shift + Enter`). GPU 없이 CPU만으로 몇 초 안에 끝난다.\n"
        "`assert`가 하나라도 실패하면 그 셀에서 오류가 나고, 모두 통과하면 마지막에 \"모든 검사를 통과했다.\"가 출력된다."
    )
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [{"cell_type": "markdown", "id": "intro", "metadata": {}, "source": to_ipynb_lines(intro)}]
        + [
            {"cell_type": "code", "id": f"cell{i}", "metadata": {}, "execution_count": None,
             "outputs": [], "source": to_ipynb_lines(c)}
            for i, c in enumerate(code_cells)
        ],
    }
    out = demo.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")

    # 검사: 셀을 하나의 네임스페이스에서 순서대로 실행한 출력이 output.txt와 같아야 한다
    ns: dict = {"__name__": "__main__"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for c in code_cells:
            exec(compile(c, f"<{slug} cell>", "exec"), ns)
    expected = (demo.parent / "output.txt").read_text()
    if buf.getvalue() != expected:
        sys.exit(f"FAIL: 노트북 실행 결과가 code/{slug}/output.txt와 다르다")
    print(f"OK: {out.relative_to(ROOT)} ({len(code_cells)}개 코드 셀, 출력 일치)")
    print(f"Colab: https://colab.research.google.com/github/NayeonS2/NayeonS2.github.io/blob/main/code/{slug}/demo.ipynb")


if __name__ == "__main__":
    main()
