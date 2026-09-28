#!/usr/bin/env bash
# Build both deliverables in submission/ from the current repo state:
#   team_solution_submission.zip            -- package in the layout the organizers require
#   matching_results_leaderboard_upload.zip -- matching_results.tsv only, for the portal
set -euo pipefail
cd "$(dirname "$0")/.."

for f in output/matching_results.tsv output/candidate_pairs.tsv; do
    [[ -f $f ]] || { echo "missing $f -- run scripts/run_pipeline.sh first" >&2; exit 1; }
done

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
pkg="$stage/code/business_entity_resolution"
mkdir -p "$stage/output" "$pkg"
cp output/matching_results.tsv output/candidate_pairs.tsv "$stage/output/"
cp -r src "$pkg/src"
find "$pkg" -name __pycache__ -prune -exec rm -rf {} +
cp README.md "$pkg/README.md"
cp requirements.txt "$pkg/requirements.txt"
cp docs/methodology.md "$stage/Documentation_template.md"

mkdir -p submission
rm -f submission/team_solution_submission.zip submission/matching_results_leaderboard_upload.zip
(cd "$stage" && zip -rq -6 "$OLDPWD/submission/team_solution_submission.zip" output code Documentation_template.md)
(cd output && zip -q -9 ../submission/matching_results_leaderboard_upload.zip matching_results.tsv)

unzip -l submission/team_solution_submission.zip
ls -lh submission/
