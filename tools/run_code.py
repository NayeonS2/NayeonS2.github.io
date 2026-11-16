"""글에 실린 예제 코드가 실제로 실행되고, 글 안의 코드·출력이 원본과 같은지 검사한다.

사용법:
    .venv/bin/python tools/run_code.py <slug>            # 실행하고 출력을 code/<slug>/output.txt에 저장
    .venv/bin/python tools/run_code.py <slug> --check    # 위 실행 + _drafts/_posts 글과 대조
    .venv/bin/python tools/run_code.py <slug> --sync     # 위 실행 후 글 안의 코드·출력 블록을 실제 내용으로 갈아끼우고 대조

글 안에서는 아래 두 블록을 대조한다.
    ```python  (바로 위 줄에 <!-- demo: <slug> --> 표시)
    ```text    (바로 위 줄에 <!-- demo-output: <slug> --> 표시)
"""

import glob
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(slug: str) -> str:
    demo = ROOT / "code" / slug / "demo.py"
    proc = subprocess.run(
        [sys.executable, str(demo)], capture_output=True, text=True, timeout=120, cwd=demo.parent
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"FAIL: {demo} 실행 실패 (exit {proc.returncode})")
    (demo.parent / "output.txt").write_text(proc.stdout)
    return proc.stdout


def block_after(marker: str, text: str):
    m = re.search(re.escape(marker) + r"\s*\n```[a-z]*\n(.*?)\n```", text, re.S)
    return None if m is None else m.group(1)


def sync(slug: str, output: str) -> None:
    source = (ROOT / "code" / slug / "demo.py").read_text().rstrip("\n")
    for post in glob.glob(str(ROOT / "_drafts" / "*.md")) + glob.glob(str(ROOT / "_posts" / "*.md")):
        text = Path(post).read_text()
        if f"<!-- demo: {slug} -->" not in text:
            continue
        for marker, body in ((f"<!-- demo: {slug} -->", source), (f"<!-- demo-output: {slug} -->", output.rstrip("\n"))):
            pat = re.compile(re.escape(marker) + r"(\s*\n```[a-z]*\n)(.*?)(\n```)", re.S)
            text, n = pat.subn(lambda m: marker + m.group(1) + body + m.group(3), text)
            if n != 1:
                sys.exit(f"FAIL: {post}에서 '{marker}' 블록을 찾지 못했어요")
        Path(post).write_text(text)
        print(f"synced: {Path(post).name}")


def check(slug: str, output: str) -> None:
    posts = [p for p in glob.glob(str(ROOT / "_drafts" / "*.md")) + glob.glob(str(ROOT / "_posts" / "*.md"))
             if f"<!-- demo: {slug} -->" in Path(p).read_text()]
    if not posts:
        sys.exit(f"FAIL: '<!-- demo: {slug} -->' 표시가 있는 글을 찾지 못했어요")
    source = (ROOT / "code" / slug / "demo.py").read_text().rstrip("\n")
    for post in posts:
        text = Path(post).read_text()
        if block_after(f"<!-- demo: {slug} -->", text) != source:
            sys.exit(f"FAIL: {post} 안의 코드가 code/{slug}/demo.py와 달라요")
        if block_after(f"<!-- demo-output: {slug} -->", text) != output.rstrip("\n"):
            sys.exit(f"FAIL: {post} 안의 실행 결과가 실제 출력과 달라요")
        print(f"OK: {Path(post).name} 코드·출력 일치")


if __name__ == "__main__":
    slug = sys.argv[1]
    out = run(slug)
    print(out, end="")
    import torch
    print(f"--- OK: code/{slug}/demo.py 실행 성공 (Python {sys.version.split()[0]}, torch {torch.__version__})")
    if "--sync" in sys.argv:
        sync(slug, out)
    if "--check" in sys.argv or "--sync" in sys.argv:
        check(slug, out)
