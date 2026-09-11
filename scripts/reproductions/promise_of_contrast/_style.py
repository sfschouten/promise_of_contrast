"""Figure sizes and fonts matched to the paper's page.

The paper uses the ACL style: a4, 2.5 cm margins, two 7.7 cm columns with 0.6 cm between
them.  Every figure is drawn at the width it is printed at, so a font size set here is the
size on the page.  Body text is 11 pt and captions 10 pt; figure text sits just below that.
"""
import matplotlib

COLUMN = 3.03    # in: one column (7.7 cm)
TEXT = 6.30      # in: the full text width (16 cm)

TICK, LABEL, LEGEND, ANNOT = 7.0, 8.0, 7.0, 6.5


def apply() -> None:
    matplotlib.rcParams.update({
        "font.size": LABEL, "axes.labelsize": LABEL, "axes.titlesize": LABEL,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": LEGEND,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
    })
