#!/usr/bin/env bash
# Run from the repo root to import all external data sources into external/.
# Each command pins the upstream commit SHA in a .dvc file next to the destination.
# Re-run `dvc update external/<name>.dvc` to bump to the latest upstream commit.
# For azaria_mitchell: `dvc update external/azaria_mitchell.zip.dvc`
#
# Estimated sizes (compressed):
#   geometry_of_truth  ~1 MB    (GitHub CSVs)
#   trilemma_of_truth  ~5 MB    (HF)
#   snli               ~100 MB  (HF)
#   arc                ~3 MB    (HF)
#   mmlu               ~5 MB    (HF; grows with more subjects in params.yaml)
#   nq_open            ~5 MB    (HF)
#   triviaqa           ~5 MB    (HF)
#   truthfulqa         ~1 MB    (HF)
#   ccs/*              ~5–500 MB each (varies; imdb/dbpedia/hellaswag are largest)
#   entailment_bank    ~20 MB   (HF)
#   azaria_mitchell    ~1 MB    (plain URL)

set -e
cd "$(dirname "$0")/.."
# A machine-local DVC binary can be set as DVC=... in an (untracked) .env file.
[ -f .env ] && source .env
DVC="${DVC:-dvc}"

mkdir -p external/ccs

# ── Active datasets ────────────────────────────────────────────────────────────

# Already imported — external/geometry_of_truth.dvc exists.
# $DVC import https://github.com/saprmarks/geometry-of-truth \
#     datasets -o external/geometry_of_truth

# Already imported — external/trilemma_of_truth.dvc exists.
# $DVC import https://huggingface.co/datasets/carlomarxx/trilemma-of-truth \
#     . -o external/trilemma_of_truth

# Already imported — external/snli.dvc exists.
# $DVC import https://huggingface.co/datasets/stanfordnlp/snli \
#     . -o external/snli

# ── Inactive datasets (wired up; activate by adding to runs/families) ──────────

# Already imported — external/arc.dvc exists.
# $DVC import https://huggingface.co/datasets/allenai/ai2_arc \
#     . -o external/arc

# Already imported — external/mmlu.dvc exists.
# $DVC import https://huggingface.co/datasets/cais/mmlu \
#     . -o external/mmlu

# Already imported — external/nq_open.dvc exists.
# $DVC import https://huggingface.co/datasets/OamPatel/iti_nq_open_val \
#     . -o external/nq_open

# Already imported — external/triviaqa.dvc exists.
# $DVC import https://huggingface.co/datasets/OamPatel/iti_trivia_qa_val \
#     . -o external/triviaqa

# Already imported — external/truthfulqa.dvc exists.
# $DVC import https://huggingface.co/datasets/truthful_qa \
#     . -o external/truthfulqa

# CCS: 11 separate HF repos, all under external/ccs/.
# rte and qnli both use the glue repo; copa uses super_glue.
$DVC import https://huggingface.co/datasets/stanfordnlp/imdb \
    . -o external/ccs/imdb

$DVC import https://huggingface.co/datasets/SetFit/amazon_polarity \
    . -o external/ccs/amazon_polarity

$DVC import https://huggingface.co/datasets/fancyzhx/ag_news \
    . -o external/ccs/ag_news

$DVC import https://huggingface.co/datasets/fancyzhx/dbpedia_14 \
    . -o external/ccs/dbpedia_14

$DVC import https://huggingface.co/datasets/nyu-mll/glue \
    . -o external/ccs/glue

$DVC import https://huggingface.co/datasets/google/boolq \
    . -o external/ccs/boolq

$DVC import https://huggingface.co/datasets/super_glue \
    . -o external/ccs/super_glue

$DVC import https://huggingface.co/datasets/Rowan/hellaswag \
    . -o external/ccs/hellaswag

$DVC import https://huggingface.co/datasets/nthngdy/piqa \
    . -o external/ccs/piqa

$DVC import https://huggingface.co/datasets/lecslab/story_cloze \
    . -o external/ccs/story_cloze

$DVC import https://huggingface.co/datasets/ariesutiono/entailment-bank-v3 \
    . -o external/entailment_bank

# azaria_mitchell: plain HTTP, not a git repo — use import-url.
# No git commit tracking; DVC uses ETag/Content-MD5 if the server provides it.
$DVC import-url http://azariaa.com/Content/Datasets/true-false-dataset.zip \
    external/azaria_mitchell.zip

echo "All imports complete."
