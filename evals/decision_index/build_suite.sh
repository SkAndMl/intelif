#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$REPO_DIR/suite-build}"
SUITE_REPO="${SUITE_REPO:-decision-index-suite-0.2}"
COMMIT=87d4650b42b377c0291a89c1f1a879f9b31082bf

if [ -f "$REPO_DIR/.env" ]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

export HF_HUB_DISABLE_XET=1

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d decision-index ]; then
    git clone --quiet https://github.com/apolinario/decision-index
fi

git -C decision-index fetch --quiet origin "$COMMIT"
git -C decision-index checkout --quiet "$COMMIT"
cd decision-index

uv venv --quiet --allow-existing --python 3.12 .venv
uv pip install --quiet --python .venv/bin/python -e ".[rebuild]"
PYTHON=.venv/bin/python

$PYTHON - <<'EOF'
import sys

from huggingface_hub import get_hf_file_metadata, hf_hub_url

try:
    get_hf_file_metadata(
        hf_hub_url(
            "cais/hle",
            "data/test-00000-of-00001.parquet",
            repo_type="dataset",
            revision="5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
        )
    )
except Exception as error:
    sys.exit(f"No access to cais/hle yet ({type(error).__name__}); request it at https://huggingface.co/datasets/cais/hle")

print("HLE access ok")
EOF

$PYTHON -m decision_index suite rebuild --work work

ROWS=work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz
ADDED=work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz

$PYTHON -m decision_index suite import --rows "$ROWS" --added-rows "$ADDED"
$PYTHON scripts/prepare_hub_upload.py --rows "$ROWS" --added-rows "$ADDED" --out hub-upload

$PYTHON - "$SUITE_REPO" <<'EOF'
import sys

from huggingface_hub import HfApi

api = HfApi()
repo_id = f"{api.whoami()['name']}/{sys.argv[1]}"
api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path="hub-upload")

print(f"uploaded to https://huggingface.co/datasets/{repo_id} (private)")
EOF
