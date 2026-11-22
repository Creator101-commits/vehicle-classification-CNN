#!/usr/bin/env bash
# Archive HEAD, version the private Kaggle source dataset, push the kernel,
# poll until it finishes, and download outputs to artifacts/kaggle/<ts>/.
# Image data comes from the private vehicle-classification-images dataset,
# which is static, so only the source dataset is
# versioned here. Kernel and dataset ids come from kernel-metadata.json.
# Usage: scripts/kaggle_run.sh [--smoke]
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

KAGGLE_BIN="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
SMOKE=0
if [ "${1:-}" = "--smoke" ]; then
  SMOKE=1
fi
META="${KAGGLE_META:-}"
if [ -z "$META" ]; then
  if [ "$SMOKE" -eq 1 ]; then
    META="kaggle_smoke/kernel-metadata.json"
  else
    META="kaggle/kernel-metadata.json"
  fi
fi
KERNEL_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$META")"
SOURCES="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["dataset_sources"]))' "$META")"
SRC_DATASET="$(echo "$SOURCES" | awk '{print $1}')"
SRC_TITLE="$(basename "$SRC_DATASET")"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="artifacts/kaggle/${TS}"

SHA="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo "warning: working tree is dirty; the dataset reflects HEAD ($SHA) only" >&2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- Source dataset: git archive of HEAD. ---
mkdir -p "$TMP/ds"
git archive HEAD | tar -x -C "$TMP/ds"

cat > "$TMP/ds/dataset-metadata.json" <<EOF
{"title": "${SRC_TITLE}", "id": "${SRC_DATASET}", "licenses": [{"name": "other"}]}
EOF
cat > "$TMP/ds/run_info.json" <<EOF
{"git_sha": "${SHA}", "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

if "$KAGGLE_BIN" datasets status "$SRC_DATASET" >/dev/null 2>&1; then
  echo "versioning existing dataset $SRC_DATASET"
  "$KAGGLE_BIN" datasets version -p "$TMP/ds" -m "src ${SHA}" -r tar -q
else
  echo "creating dataset $SRC_DATASET"
  "$KAGGLE_BIN" datasets create -p "$TMP/ds" -r tar -q
fi

# Wait for the dataset version to finish processing.
READY=0
for _ in $(seq 1 60); do
  STATUS="$("$KAGGLE_BIN" datasets status "$SRC_DATASET" 2>/dev/null || echo unknown)"
  echo "dataset $SRC_DATASET status: $STATUS"
  case "$STATUS" in
    *ready*|*complete*) READY=1; break ;;
  esac
  sleep 10
done
if [ "$READY" -ne 1 ]; then
  echo "error: dataset $SRC_DATASET did not become ready" >&2
  exit 1
fi

# "ready" precedes mount propagation by minutes; kernels pushed too early
# mount the previous dataset version and fail the git_sha guard.
sleep 300

echo "pushing kernel $KERNEL_ID"
PUSH_DIR="$(dirname "$META")"
"$KAGGLE_BIN" kernels push -p "$PUSH_DIR"

# Poll kernel status until a terminal state; bounded by the loop count.
TERMINAL=""
for _ in $(seq 1 720); do
  STATUS="$("$KAGGLE_BIN" kernels status "$KERNEL_ID" 2>&1 || echo "status-poll-error")"
  echo "kernel status: $STATUS"
  case "$STATUS" in
    *COMPLETE*|*ERROR*|*CANCEL*) TERMINAL="$STATUS"; break ;;
  esac
  sleep 60
done

mkdir -p "$OUT_DIR"
echo "downloading kernel output to $OUT_DIR"
"$KAGGLE_BIN" kernels output "$KERNEL_ID" -p "$OUT_DIR"

# Guard against the dataset-version race: the run must reflect HEAD.
MANIFEST="$OUT_DIR/benchmark/manifest.json"
if [ -f "$MANIFEST" ]; then
  RUN_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("git_sha",""))' "$MANIFEST")"
  if [ "$RUN_SHA" != "$SHA" ]; then
    echo "error: run manifest git_sha=$RUN_SHA does not match HEAD $SHA" >&2
    exit 1
  fi
fi

case "$TERMINAL" in
  *COMPLETE*) ;;
  *)
    echo "error: kernel did not complete (status: ${TERMINAL:-poll-timeout})" >&2
    exit 1
    ;;
esac

echo "done: kernel=$KERNEL_ID sha=$SHA out=$OUT_DIR"
