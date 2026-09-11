"""Accuracy tables for the Probing Without Labels reproduction.

Everything is read from the per-run DuckDB written by `evaluate_probes.py`, and every
number is the **uncalibrated** accuracy: threshold (p_base + 1 - p_cf)/2 at 0.5 against
the sample-level truth, with no calibration head and no sign disambiguation.  That is the
quantity unsupervised contrast probing reports, and it is why values below 0.5 appear and
are left as they are.

Probes are always scored at the position they were trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ABLATION_DATASETS, DATASETS, POSITION_LABELS

METRIC = "uncalibrated_accuracy"


def load_evals(db_path) -> pd.DataFrame:
    import duckdb

    conn = duckdb.connect(str(db_path), read_only=True)
    df = conn.execute(f"""
        SELECT dataset, layer, train_position, eval_position, method, seed,
               {METRIC} AS acc, accuracy AS calibrated_accuracy, auc, n_eval,
               pair_accuracy, pos_prob_mean, neg_prob_mean, sign_flipped
        FROM probe_evals
        WHERE train_position = eval_position
    """).fetch_df()
    conn.close()
    if df.empty:
        return df
    df["position"] = df["eval_position"].map(POSITION_LABELS).fillna(df["eval_position"])
    return df


def table_sign_instability(df: pd.DataFrame, catalogue: dict) -> pd.DataFrame:
    """How often the unsupervised orientation came out backwards, per dataset.

    The objectives are invariant under theta -> -theta, so this is a property of the
    method rather than a defect: a value near 50% means the axis is found reliably but its
    direction is a coin flip across seeds.
    """
    ccs = {m for m, i in catalogue.items() if i["kind"] == "ccs"}
    sub = df[df["method"].isin(ccs) & df["sign_flipped"].notna()]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for ds in _ordered(sub, DATASETS):
        row = {"dataset": ds}
        for pos in ("answer", "period"):
            cell = sub[(sub["dataset"] == ds) & (sub["position"] == pos)]
            row[f"{pos} flipped (%)"] = (
                f"{100 * cell['sign_flipped'].mean():.0f}" if len(cell) else "—")
        rows.append(row)
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return "—" if x is None or np.isnan(x) else f"{100 * x:.0f}"


def _mean_sd(values: pd.Series) -> str:
    """The paper's "mean ± sd" cell, both as integer percentages."""
    if values.empty or values.isna().all():
        return "—"
    return f"{_pct(values.mean())} ± {100 * values.std(ddof=0):02.0f}"


def _ordered(df: pd.DataFrame, names: list[str]) -> list[str]:
    present = set(df["dataset"])
    return [n for n in names if n in present] + sorted(present - set(names))


def table_ccs_accuracy(df: pd.DataFrame, catalogue: dict) -> pd.DataFrame:
    """Table 1 — CCS accuracy per dataset, at both token positions, over all seeds."""
    method = _method_for(catalogue, "CCS")
    sub = df[df["method"] == method]
    rows = []
    for ds in _ordered(sub, DATASETS):
        row = {"dataset": ds}
        for pos in ("answer", "period"):
            cell = sub[(sub["dataset"] == ds) & (sub["position"] == pos)]
            row[f"{pos} (%)"] = _mean_sd(cell["acc"])
            row[f"n_{pos}"] = int(cell["seed"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def table_loss_ablations(df: pd.DataFrame, catalogue: dict,
                         position: str = "answer") -> pd.DataFrame:
    """Table 2 — the seven CCS objectives, on the datasets where CCS performs well."""
    order = ["CCS", "L_conf", "L_cons", "L_cons+a1", "L_cons+a2",
             "L_cons+a1+a2", "CCS+a1+a2"]
    sub = df[df["position"] == position]
    rows = []
    for ds in _ordered(sub[sub["dataset"].isin(ABLATION_DATASETS)], ABLATION_DATASETS):
        row = {"dataset": ds}
        for label in order:
            method = _method_for(catalogue, label, required=False)
            cell = sub[(sub["dataset"] == ds) & (sub["method"] == method)] if method else sub.iloc[:0]
            row[label] = _mean_sd(cell["acc"])
        rows.append(row)
    return pd.DataFrame(rows)


def table_tcpca_vs_ccs(df: pd.DataFrame, catalogue: dict,
                       position: str = "period") -> pd.DataFrame:
    """Table 3 — tcPCA against the spread of CCS solutions, and against CRC-TPC.

    CCS is summarised by min/median/max over seeds rather than mean ± sd: the point is
    that its outcome depends on the initialisation, and a mean hides that.  The whitened
    variant (GCC) gets its own column when its probes exist.
    """
    ccs = _method_for(catalogue, "CCS")
    crc = _method_for(catalogue, "PC1", required=False)
    tcpca = _method_for(catalogue, "tcPCA-1", required=False)
    gcc = _method_for(catalogue, "GCC-1", required=False)
    sub = df[df["position"] == position]

    rows = []
    for ds in _ordered(sub, DATASETS):
        at = sub[sub["dataset"] == ds]
        seeds = at[at["method"] == ccs]["acc"]
        row = {
            "dataset": ds,
            "CCS min": _pct(seeds.min()) if len(seeds) else "—",
            "CCS med": _pct(seeds.median()) if len(seeds) else "—",
            "CCS max": _pct(seeds.max()) if len(seeds) else "—",
        }
        for label, method in (("CRC-TPC", crc), ("tcPCA", tcpca), ("GCC", gcc)):
            vals = at[at["method"] == method]["acc"] if method else at.iloc[:0]["acc"]
            row[label] = _pct(vals.iloc[0]) if len(vals) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def table_principal_components(df: pd.DataFrame, catalogue: dict) -> pd.DataFrame:
    """Table 5 — accuracy of each of the first five principal components of X_cf - X_base."""
    labels = ["PC1", "PC2", "PC3", "PC4", "PC5"]
    rows = []
    for pos in ("answer", "period"):
        sub = df[df["position"] == pos]
        for ds in _ordered(sub, DATASETS):
            row = {"dataset": ds, "token": pos}
            for label in labels:
                method = _method_for(catalogue, label, required=False)
                vals = sub[(sub["dataset"] == ds) & (sub["method"] == method)]["acc"] \
                    if method else sub.iloc[:0]["acc"]
                row[label] = _pct(vals.iloc[0]) if len(vals) else "—"
            rows.append(row)
    return pd.DataFrame(rows)


def _method_for(catalogue: dict, label: str, required: bool = True) -> str | None:
    for method, info in catalogue.items():
        if info["label"] == label:
            return method
    if required:
        raise KeyError(f"no configured method labelled {label!r}")
    return None


# ── LaTeX versions, \input by the paper so no number is copied by hand ────────
#
# Each writer emits a complete tabular (the paper keeps its own table environment,
# caption and label around the \input).  Macros such as \c, \mc, \mr, \bf and \wtcpca are
# the paper's own, defined where it \inputs these files.

def _tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def _pm(cell: str, tight: bool = False) -> str:
    """"93 ± 01" -> "$ 93 \\pm 01$" (or the tighter ${\\pm}$ form Table 2 uses)."""
    if not isinstance(cell, str) or "±" not in cell:
        return "—" if cell in (None, "—") else str(cell)
    mean, sd = (p.strip() for p in cell.split("±"))
    return rf"${mean:>3}{{\pm}}{sd}$" if tight else rf"${mean:>3} \pm {sd}$"


def latex_table1(t1: pd.DataFrame) -> str:
    rows = [rf"        {_tex(r['dataset']):<15} & {_pm(r['answer (%)'])} & {_pm(r['period (%)'])} \\"
            for _, r in t1.iterrows()]
    return "\n".join([
        r"    \begin{tabular}{lrr}", r"        \toprule",
        r"        Dataset         &\textit{answer} (\%)&\textit{period} (\%) \\",
        r"        \midrule", *rows, r"        \bottomrule", r"    \end{tabular}", ""])


TABLE2_GROUPS = [(r"\citet{marks_geometry_2024}", ["comparisons", "sp_en_trans", "cities"]),
                 (r"\citet{burns_discovering_2023}", ["amazon", "imdb"])]
TABLE2_COLS = ["CCS", "L_conf", "L_cons", "L_cons+a1", "L_cons+a2", "L_cons+a1+a2", "CCS+a1+a2"]


def latex_table2(t2: pd.DataFrame) -> str:
    by_ds = {r["dataset"]: r for _, r in t2.iterrows()}
    out = [r"    \begin{tabular}{p{3.5cm}lrrrrrrr}", r"    \toprule",
           r"    $\mu \pm \sigma$ (\%) && \c{CCS} & \c{$\L{conf}$} & \c{$\L{cons}$} & \c{$\L{cons}$} & \c{$\L{cons}$} & \c{$\L{cons}$} & \c{CCS}  \\",
           r"    && \c{-} & \c{-} & \c{-} & \c{+a1} & \c{+a2} & \c{+a1+a2} & \c{+a1+a2}\\",
           r"    \midrule"]
    for g, (source, names) in enumerate(TABLE2_GROUPS):
        present = [n for n in names if n in by_ds]
        if not present:
            continue
        if g:
            out.append(r"    \cmidrule{1-9}")
        for i, ds in enumerate(present):
            lead = rf"\mr{{{len(present)}}}{{3.5cm}}{{{source}}}" if i == 0 else ""
            cells = " & ".join(_pm(by_ds[ds].get(c, "—"), tight=True) for c in TABLE2_COLS)
            out.append(rf"    {lead}" + "\n" + rf"        & {_tex(ds):<16} & {cells} \\"
                       if i == 0 else rf"        & {_tex(ds):<16} & {cells} \\")
    out += [r"    \bottomrule", r"    \end{tabular}", ""]
    return "\n".join(out)


def latex_table3(t3: pd.DataFrame) -> str:
    """Table 3 with bold where a method matches or exceeds both the CCS median and CRC-TPC."""
    has_w = "GCC" in t3.columns
    num = lambda v: float(v) if str(v).replace(".", "").isdigit() else float("nan")
    head_extra = r" & \mr{2.2}{*}{\wtcpca}" if has_w else ""
    out = [r"\begin{tabular}{lrrrrr" + ("r" if has_w else "") + "}", r"\toprule",
           r"               & \mc{3}{c}{CCS} & CRC & \mr{2.2}{*}{tcPCA}" + head_extra + r" \\",
           r"Dataset        & min & med & max & -TPC &" + (" &" if has_w else "") + r" \\ \midrule"]
    for _, r in t3.iterrows():
        bar = max(num(r["CCS med"]), num(r["CRC-TPC"]))
        cell = lambda v: (r"\bf " if num(v) >= bar else "") + str(v)
        extra = f" & {cell(r['GCC'])}" if has_w else ""
        out.append(rf"{_tex(r['dataset']):<14} & {r['CCS min']} & {r['CCS med']} & {r['CCS max']} & "
                   rf"{r['CRC-TPC']} & {cell(r['tcPCA'])}{extra} \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def latex_table5(t5: pd.DataFrame) -> str:
    pcs = [c for c in t5.columns if c.startswith("PC")]
    out = [r"    \begin{tabular}{ll" + "r" * len(pcs) + "}", r"    \toprule",
           "    Dataset & token & " + " & ".join(pcs) + r" \\", r"    \midrule"]
    last = None
    for _, r in t5.iterrows():
        if last is not None and r["token"] != last:
            out.append(r"    \midrule")
        last = r["token"]
        out.append(f"    {_tex(r['dataset'])} & \\textit{{{r['token']}}} & "
                   + " & ".join(str(r[c]) for c in pcs) + r" \\")
    out += [r"    \bottomrule", r"    \end{tabular}", ""]
    return "\n".join(out)


def protocol_rows(params: dict, profile: str, alias: str) -> list[tuple[str, str]]:
    """The reproduction protocol as (setting, value) rows, read from params.yaml."""
    pp = {**params.get("probing", {}), **params["probing_profiles"][profile]}
    ap, mp = params["activations"], params["models"][alias]
    ccs = next(m for m in pp["methods"] if m["name"] == "ccs")
    layers = pp.get("layers", ap["layers"])
    return [
        ("Model", rf"\texttt{{{_tex(mp['name'])}}}, {mp.get('dtype', 'auto')}, batch {mp['batch_size']}"),
        ("Layer", "output of decoder block " + ", ".join(str(l) for l in layers)
                  + r" (\texttt{hidden\_states[" + ", ".join(str(l + 1) for l in layers) + "]})"),
        ("Tokens", r"\textit{answer} = second-to-last, \textit{period} = last"),
        ("Split", f"{int(100 * pp['train_split'])}/{100 - int(100 * pp['train_split'])} over pairs, "
                  f"{pp.get('split_strategy', 'random')} (fixed across seeds)"),
        ("Centring", _tex(pp.get("centering", "midpoint")).replace(r"per\_branch", "per branch (each of $X^+$, $X^-$ separately)")),
        ("CCS", f"{ccs.get('optimizer', 'adam')}, lr {ccs['lr']}, weight decay {ccs['weight_decay']}, "
                f"{ccs['n_epochs']} epochs, {ccs.get('n_restarts', 1)} restart, "
                f"{len(ccs.get('seeds') or [])} seeds; no variance standardisation"
                if not ccs.get("standardize", True) else
                f"{ccs.get('optimizer', 'adam')}, lr {ccs['lr']}, {ccs['n_epochs']} epochs, "
                f"{len(ccs.get('seeds') or [])} seeds; standardised by pooled std"),
        ("tcPCA", r"eigenvectors of $\mathrm{cov}(X^+{+}X^-) - N\,\mathrm{cov}(X^*)$, exact"),
        (r"\wtcpca", r"same, after whitening by $\mathrm{cov}(X^*)$ (ridge $10^{-8}\|\Sigma\|_F$)"),
        ("Orientation", _tex(pp.get("sign_from", "none")).replace(r"diff\_in\_means",
                        r"aligned with $\bar{x}_\mathrm{true} - \bar{x}_\mathrm{false}$ on the train split")),
        ("Metric", r"$\tfrac12(p(x^+) + 1 - p(x^-)) > 0.5$ on held-out pairs, no calibration"),
    ]


def latex_protocol(rows: list[tuple[str, str]]) -> str:
    body = [rf"    {k} & {v} \\" for k, v in rows]
    return "\n".join([r"    \begin{tabular}{lp{0.72\linewidth}}", r"    \toprule",
                      *body, r"    \bottomrule", r"    \end{tabular}", ""])


def latex_crc_variance(df: pd.DataFrame) -> str:
    """Loudness / purity / accuracy of the CRC-TPC, tcPCA and g-tcPCA directions."""
    methods = [m for m in ("CRC-TPC", "tcPCA", "g-tcPCA") if m in set(df["method"])]
    shown = {"g-tcPCA": r"\wtcpca"}
    head = " & ".join(rf"\mc{{3}}{{c}}{{{shown.get(m, m)}}}" for m in methods)
    sub = " & ".join(["loud & pure & acc"] * len(methods))
    rules = " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(methods)))
    out = [r"    \begin{tabular}{ll" + "rrr" * len(methods) + "}", r"    \toprule",
           rf"    & & {head} \\", rf"    {rules}", rf"    Dataset & token & {sub} \\", r"    \midrule"]
    last = None
    for position in ("answer", "period"):
        for ds in [d for d in DATASETS if d in set(df["dataset"])]:
            at = df[(df["dataset"] == ds) & (df["position"] == position)]
            if at.empty:
                continue
            if last is not None and position != last:
                out.append(r"    \midrule")
            last = position
            cells = []
            for m in methods:
                r = at[at["method"] == m]
                cells += ([f"{r['loudness'].iloc[0]:.2f}", f"{r['purity'].iloc[0]:.2f}",
                           f"{100 * r['accuracy'].iloc[0]:.0f}"] if len(r) else ["—"] * 3)
            out.append(rf"    {_tex(ds)} & \textit{{{position}}} & " + " & ".join(cells) + r" \\")
    out += [r"    \bottomrule", r"    \end{tabular}", ""]
    return "\n".join(out)


def latex_copa_components(df: pd.DataFrame) -> str:
    """COPA: accuracy and eigenvalue of each of the first tcPCs, per token position."""
    k = int(df["component"].max())
    out = [r"    \begin{tabular}{ll" + "r" * k + "}", r"    \toprule",
           "    token & & " + " & ".join(f"tcPC {i}" for i in range(1, k + 1)) + r" \\",
           r"    \midrule"]
    for position in ("answer", "period"):
        at = df[df["position"] == position].sort_values("component")
        if at.empty:
            continue
        out.append(rf"    \textit{{{position}}} & acc (\%) & "
                   + " & ".join(f"{100 * a:.0f}" for a in at["accuracy"]) + r" \\")
        out.append(r"     & $\lambda$ & " + " & ".join(f"{e:.2f}" for e in at["eigenvalue"]) + r" \\")
    out += [r"    \bottomrule", r"    \end{tabular}", ""]
    return "\n".join(out)


def _tex_text(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def latex_copa_examples_full(pairs: pd.DataFrame, n_top: int = 10, n_low: int = 6) -> str:
    """Appendix table: the most contrastive prompts, then prompts the direction barely separates.

    Each row gives the higher and the lower of the two choices' activations; the choice with the
    higher activation is set in bold.
    """
    def row(r, bold: bool) -> str:
        hi = 1 if r["a1"] >= r["a2"] else 2
        c = {1: _tex_text(r["choice1"]), 2: _tex_text(r["choice2"])}
        if bold:
            c[hi] = rf"\textbf{{{c[hi]}}}"
        q = _tex_text(r["question"].split(" Choice 1:")[0])
        tail = _tex_text(r["question"].split(" Q:")[-1])
        prompt = f"{q} Choice 1: {c[1]} Choice 2: {c[2]} Q:{tail} choice [1/2]."
        return f" {max(r['a1'], r['a2']):5.2f}  &  {min(r['a1'], r['a2']):5.2f}    &    {prompt}\\\\"

    top = pairs.sort_values("gap", ascending=False).head(n_top)
    low = pairs[~pairs.index.isin(top.index)].sort_values("peak").head(n_low)
    out = [r"\begin{tabular}{rrp{11cm}}", r"\toprule",
           r"\multicolumn{2}{c}{Act. strengths} & Prompt \\", r"\midrule"]
    out += [row(r, True) for _, r in top.iterrows()]
    out += [r"\midrule"] + [row(r, False) for _, r in low.iterrows()]
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def latex_copa_examples_main(pairs: pd.DataFrame, per_kind: int = 2) -> str:
    """Main-text table: per question kind, the most contrastive prompts.  Left is the choice with
    the higher activation, right the lower; the commonsense (correct) choice is underlined."""
    out = [r"\begin{tabular}{lp{5.2cm}rlrl}", r"    \toprule",
           r"    & prompt & \mc{1}{c}{$a$} & negative sentiment & \mc{1}{c}{$a$} & positive sentiment \\",
           r"\midrule"]
    for i, kind in enumerate(("effect", "cause")):
        sel = pairs[pairs["kind"] == kind].sort_values("gap", ascending=False).head(per_kind)
        if sel.empty:
            continue
        if i:
            out.append(r"\midrule")
        for j, (_, r) in enumerate(sel.iterrows()):
            hi, lo = (1, 2) if r["a1"] >= r["a2"] else (2, 1)
            text = {1: _tex_text(r["choice1"]), 2: _tex_text(r["choice2"])}
            act = {1: r["a1"], 2: r["a2"]}
            fmt = lambda k: rf"\uline{{`{text[k]}'}}" if r["correct"] == k else f"`{text[k]}'"
            lead = (rf"    \mr{{{len(sel)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\underline{{{kind}}}}}}}"
                    if j == 0 else "   ")
            out.append(f"{lead}\n    & `{_tex_text(r['premise'])}'\n"
                       f"    & {act[hi]:4.1f} & {fmt(hi)}\n    & {act[lo]:4.1f} & {fmt(lo)} \\\\")
    out += [r"    \bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)
