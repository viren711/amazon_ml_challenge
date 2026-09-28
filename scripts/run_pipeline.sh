#!/usr/bin/env bash
# End-to-end run: train -> predict -> validate.
#   scripts/run_pipeline.sh [all|train|predict|validate]
# Override the dataset location with DATA=/path/to/dataset (must contain train/ and test/).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-student_resource/dataset}
STAGE=${1:-all}

train() {
    python3 src/er.py train --data-dir "$DATA/train" --out-dir artifacts
}

predict() {
    python3 src/er.py predict --data-dir "$DATA/test" --artifacts-dir artifacts --out-dir output
}

validate() {
    python3 student_resource/utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir "$DATA/test" --check-ids
}

case "$STAGE" in
    all) train; predict; validate ;;
    train) train ;;
    predict) predict ;;
    validate) validate ;;
    *) echo "usage: $0 [all|train|predict|validate]" >&2; exit 2 ;;
esac
