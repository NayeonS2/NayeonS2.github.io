#!/usr/bin/env bash
# 초안(_drafts)까지 포함해서 로컬 미리보기 서버를 띄운다.
# 사용법: bash tools/serve-drafts.sh   →  http://127.0.0.1:4000
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/opt/ruby@3.4/bin:$PATH"
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export RUBYOPT="-EUTF-8:UTF-8"
# --future: 예약 날짜(미래 날짜)가 적힌 초안도 미리보기에 보이게 한다
# --force_polling: 한글 경로에서는 파일 변경 감지(fsevents)가 동작하지 않아서 주기적으로 확인한다
exec bundle exec jekyll serve --drafts --future --force_polling --host 127.0.0.1 --port 4000
