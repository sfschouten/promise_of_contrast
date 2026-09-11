"""Per-dataset construction helpers.

Everything else in `src/data/` is dataset-agnostic — `Sample`, `ContrastivePair`, `Graph`,
templating, masks.  These modules are not: each one knows the shape of one specific source
(the GeoNames city dump, a cyclic weekday vocabulary, EntailmentBank's proof trees) and
exists only to be called by the matching generator in `scripts/`.

They are kept out of the general layer so that `src/data/` stays readable as the data model
rather than as a pile of dataset trivia.  Nothing here is re-exported from `src.data`:
import the module you need directly.
"""
