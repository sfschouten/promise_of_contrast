import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass
class Sample:
    id: str
    text: str
    descriptors: dict[str, Any] = field(default_factory=dict)
    # Character-level spans: each value is a list of (start, end) pairs (exclusive end).
    masks: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> Any:
        return self.descriptors.get("label")

    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "descriptors": self.descriptors,
            "masks": self.masks,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        # Backward compat: old format stored label at top level.
        descriptors = dict(d.get("descriptors", {}))
        if "label" not in descriptors and "label" in d:
            descriptors["label"] = d["label"]
        # JSON round-trips tuples as lists; restore as tuples.
        masks = {
            k: [tuple(span) for span in spans]
            for k, spans in d.get("masks", {}).items()
        }
        return cls(
            id=d["id"],
            text=d["text"],
            descriptors=descriptors,
            masks=masks,
            metadata=d.get("metadata", {}),
        )


def load_hf_dataset(
    name: str,
    split: str,
    text_field: str,
    label_field: str,
    max_samples: int | None = None,
    **kwargs,
) -> list[Sample]:
    ds = load_dataset(name, split=split, **kwargs)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return [
        Sample(
            id=str(i),
            text=row[text_field],
            descriptors={"label": row[label_field]},
            metadata={k: v for k, v in row.items() if k not in (text_field, label_field)},
        )
        for i, row in enumerate(ds)
    ]


def load_jsonl(path: str) -> list[Sample]:
    with open(path) as f:
        return [Sample.from_dict(json.loads(line)) for line in f]


def save_jsonl(samples: list[Sample], path: str) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")
