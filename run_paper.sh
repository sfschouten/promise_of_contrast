#!/usr/bin/env bash
# Reproduce "Probing Without Labels" end to end, on both models.
#
# The expensive step is activation extraction (two 7-8B models over ~15,700 prompts);
# everything after it is cheap and cached, so re-running this after a code change costs
# minutes rather than hours.
#
# Needs a GPU with ~16 GB free, and a HuggingFace login with access to the two gated
# Llama checkpoints.  Run from the project root.

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Long prompts fragment the allocator; expandable segments keeps peak memory down.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== data =="
# The datasets are DVC imports pinned to upstream revisions; there is no DVC remote, so
# `dvc pull` fetches them from their origins.  A no-op once they are present.
dvc pull external/*.dvc external/ccs/*.dvc
dvc repro generate@pwl make_pairs@pwl preprocess@pwl

for model in llama2_7b llama3_8b; do
  echo "== ${model} =="
  dvc repro "extract_activations@${model}__pwl"
  dvc repro "train_probes@${model}__pwl"
  dvc repro "evaluate_probes@${model}__pwl"
  dvc repro "reproduce_promise_of_contrast@${model}"
done

dvc repro --single-item check_paper

echo "== compare against the committed results =="
python scripts/reproductions/promise_of_contrast/compare_to_committed.py

echo
echo "Tables and figures: outputs/reproductions/promise_of_contrast/<model>/"
