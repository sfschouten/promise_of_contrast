# Probing Without Labels

Code for *Probing Without Labels: Contrast-only Reproductions of Interpretability
Results*, plus the general contrastive-probing pipeline it is built on.

The central object is a **contrastive pair**: two inputs that differ in one controlled
way.

---

## Getting started

```bash
mamba env create -f environment.yml -p ./env   # or conda >= 23.10; or --name interp
conda activate ./env
huggingface-cli login                          # the Llama checkpoints are gated
./run_paper.sh                                 # fetch the data, build the paper
```

Both models are gated on HuggingFace: accept the licences for
[`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) and
[`meta-llama/Meta-Llama-3.1-8B`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
before the first run.

Activation extraction needs a GPU with room for a 7–8B model in half precision (~16 GB).
Everything downstream runs on CPU.

The older conda solver (before 23.10) can exhaust memory solving this environment; use
mamba, or a conda recent enough to default to the libmamba solver.

Prefer `./run_paper.sh` (or `dvc repro check_paper`) over a bare `dvc repro`: the latter
builds every stage in `dvc.yaml`, including the general-purpose reports, which the paper
does not need.

---

## The pipeline

```
generate_*  →  [make_pairs]  →  preprocess  →  extract_activations
                                                     ↓
             report_* / reproduce_*  ←  evaluate_probes  ←  train_probes
```

Stages live in `dvc.yaml`, configuration in `params.yaml`. Most stages are `foreach`
loops over a section of `params.yaml`, so adding a dataset or a model is a config change
rather than a code change.

| Stage | What it does |
|---|---|
| `generate@<dataset>` | `scripts/generate_<dataset>.py` → `data/raw/<dataset>.jsonl` |
| `make_pairs@<family>` | builds contrastive pairs from flat samples (skipped for natively paired data) |
| `preprocess@<family>` | applies a Jinja template → `data/processed/samples_<family>.jsonl` |
| `extract_activations@<run>` | residual-stream activations via **nnsight**, cached per sample |
| `train_probes@<run>` | fits every configured probe at every (layer, position, seed) |
| `evaluate_probes@<run>` | scores probes on their held-out pairs → DuckDB |

A **run** pairs a model with a data family (`llama2_7b__pwl`); a **compound** pools
several sub-datasets into one training set.

### Two conventions worth knowing

**Layer numbering is block numbering.** `activations.layers: 15` is the output of decoder
block 15, i.e. `outputs.hidden_states[16]`, i.e. `model.model.layers[15].output`. The
embedding layer is not addressable this way.

**Pairs are ordered by framing, not by truth.** `base` is branch 0 and `counterfactual` is
branch 1, fixed by the template; which one is *correct* is carried separately in the
`truth` descriptor. If `base` were always the true side, an unsupervised probe could score
100% by learning the ordering, and below-chance accuracy — a real and informative outcome
— could not occur.

---

## Reproducing the paper

`run_paper.sh` does the whole thing end to end, on both models. Step by step:

```bash
dvc pull external/*.dvc external/ccs/*.dvc         # fetch the pinned datasets (~2.5 GB)
dvc repro reproduce_promise_of_contrast@llama2_7b  # tables and figures
dvc repro reproduce_promise_of_contrast@llama3_8b
```

Outputs land in `outputs/reproductions/promise_of_contrast/<model>/`, and that directory
is **committed** — the tables, the figures and the per-seed CSVs are in this repository, so
they can be read without rebuilding anything. Every experiment is run on **both**
Llama-2-7B and Llama-3.1-8B.

`check_paper.py` is the build target, and it is a DVC stage:

```bash
dvc repro check_paper       # every table and figure present, for both models
```

It depends on both reproduction outputs, so it pulls in exactly the paper's chain and
nothing auxiliary, and it fails loudly rather than producing a reassuring file nobody
reads.

A rebuild overwrites the committed results in place, so git still holds the originals.
`compare_to_committed.py` (the last step of `run_paper.sh`) compares the two: the example
pools must be identical, the paper's tables must agree within 3 points (or twice a cell's
own seed spread), and every other CSV is reported. On the same GPU the numbers reproduce
exactly; other hardware perturbs bf16 activations slightly, hence the tolerance.

```bash
python scripts/reproductions/promise_of_contrast/compare_to_committed.py
```

> **There is no DVC remote**, and `dvc.lock` is not published — on a fresh clone there
> are no artifacts for it to vouch for. The pipeline rebuilds from source:
> `external/*.dvc` are DVC *imports* pinned to upstream revisions, so `dvc pull` fetches
> the datasets from their origins (GitHub and HuggingFace), and everything after that is
> computed. The results themselves are committed, so the tables and figures can be read
> without running anything.

---

## Probes (`src/probing/`)

| Probe | Method name | Notes |
|---|---|---|
| `LogisticProbe` | `logistic` | supervised reference point |
| `DiffInMeansProbe` | `diff_in_means` | supervised mass-mean probe, `mean(true) − mean(false)` |
| `ContrastDiffInMeansProbe` | `contrast_diff_in_means` | the label-free counterpart, `mean(base) − mean(cf)` |
| `PCAProbe` | `pca_{k}` | top-k PCA + logistic head |
| `PairedPCAProbe` | `pca_pair_{style}_k{i}` | PCA of a derived matrix (e.g. `X_cf − X_base`); one component as the direction |
| `CrossCovarianceProbe` | `cross_covariance_{k}` | eigendecomposition of the symmetrised cross-covariance — the tuple-contrastive method |
| `GeneralizedCrossCovarianceProbe` | `generalized_cross_covariance_{k}` | variance-normalised variant |
| `CCSProbe` | `ccs[...]` | Contrast-Consistent Search, with loss-term ablations and search-space alterations |

A probe's `method` string encodes **only the options that differ from its defaults**, so
`ccs` and `cross_covariance_8` mean exactly what they always did while
`ccs[conf=min_sq,svd,wn]` is a distinct, self-describing variant. The string is the cache
key and the database key, so this keeps old results valid and new ones legible.

### Seed sweeps

An unsupervised probe's answer can depend on its initialisation, so a method may declare
`seeds: [0, …, 29]`. All seeds for a layer are fitted in **one batched optimisation**, and
the train/test split is held fixed across them (`split_strategy: ordered`), so the spread
that comes back is initialisation variance and nothing else.

---

## Layout

```
src/data/          Sample, ContrastivePair, Graph; templating and masks
src/data/sources/  per-dataset builders, one module per source
src/activations/   nnsight extraction and the per-sample activation cache
src/probing/       probe implementations, fitting, evaluation, metrics
src/results/       DuckDB schema and inserts
scripts/           one generator per dataset, the pipeline stages, and the reports
scripts/reproductions/<paper>/   per-paper reproduction, one DVC stage each
templates/         Jinja prompt templates
tests/             structural tests for the datasets, and unit tests for the probe algebra
```

---

## Citation

*Probing Without Labels* has been accepted at BlackboxNLP 2026. The exact citation will
follow once the proceedings are out.

---

## License

The code is released under the [MIT License](LICENSE). The datasets are fetched from their
original sources and remain under their own licences.
