#!/usr/bin/env bash
# End-to-end test of every audit layer on the shipped synthetic test data set.
# Requires: python >= 3.9, numpy, pandas, scipy, scikit-learn
#
#   bash run_test.sh
#
# Exit code 0 = the pipeline reproduces the expected results and returns the
# verdicts the test data were built to produce.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

echo "== regenerate inputs (deterministic; overwrites with identical files) =="
"$PY" make_test_data.py

echo
echo "== run every layer (L1-L5 plus the two added analyses) =="
"$PY" run_test.py

echo
echo "== verify: determinism, agreement with expected_output_*.csv, behaviour =="
"$PY" verify.py
