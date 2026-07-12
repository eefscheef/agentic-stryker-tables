#!/usr/bin/env bash
# Fetch and unpack the run data for the thesis analysis pipeline.
#
# Usage:  ./fetch-data.sh [record-url]
#
# Downloads every data tarball from the Zenodo record into .data-download/,
# verifies checksums, and extracts them under artifact/data/.
set -euo pipefail

RECORD_URL="${1:-https://zenodo.org/records/21326300}"

FILES=(
  data-cushman.tar.gz
  data-gptturbo.tar.gz
  data-starcoder.tar.gz
  data-gpt-5.4-output-500.tar.gz
  data-gpt-5.4-output-1000.tar.gz
  data-gpt-5.4-agent-pilot.tar.gz
  data-gpt-5.4-agent-residual.tar.gz
  data-gpt-5.4-agent-hidden.tar.gz
  SHA256SUMS
)

cd "$(dirname "$0")"
mkdir -p .data-download

for f in "${FILES[@]}"; do
  if [ ! -f ".data-download/$f" ]; then
    echo "downloading $f ..."
    curl -fL --retry 3 -o ".data-download/$f" "$RECORD_URL/files/$f?download=1"
  fi
done

echo "verifying checksums ..."
(cd .data-download && grep -v graceful-fs SHA256SUMS | sha256sum -c -)

echo "extracting into artifact/data/ ..."
for f in .data-download/data-*.tar.gz; do
  tar -xzf "$f" -C artifact/
done

echo "done. Run:  make all PY=<your-python>"
