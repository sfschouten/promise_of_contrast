from .base import (
    Probe,
    ProbeFitInfo,
    ProbeEvalResult,
    compute_probe_config_hash,
    fit_probe,
    fit_probe_batched,
    fit_probe_ccs_layerbatched,
    eval_probe,
    eval_probe_ood,
    eval_probe_ood_refit,
    eval_probe_transfer,
    load_probes,
    save_probes,
)
from .ccs import CCSProbe
from .crosscov import CrossCovarianceProbe, GeneralizedCrossCovarianceProbe
from .dim import ContrastDiffInMeansProbe, DiffInMeansProbe
from .linear import LogisticProbe
from .manifest import Manifest, probe_key
from .paper_eval import unsupervised_metrics
from .subspace_overlap import (
    DEFAULT_KS,
    lambda_k,
    lambda_k_from_pcs,
    max_overlap,
    max_overlap_from_pcs,
    principal_directions,
)
from .pca import PairedPCAProbe, PCAProbe


REGISTRY: dict[str, type[Probe]] = {
    "logistic":                      LogisticProbe,
    "pca":                           PCAProbe,
    "pca_pair":                      PairedPCAProbe,
    "diff_in_means":                 DiffInMeansProbe,
    "contrast_diff_in_means":        ContrastDiffInMeansProbe,
    "cross_covariance":              CrossCovarianceProbe,
    "generalized_cross_covariance":  GeneralizedCrossCovarianceProbe,
    "ccs":                           CCSProbe,
}
