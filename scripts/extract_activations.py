import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activations.extract import extract_hidden_states
from src.data.loader import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    params = yaml.safe_load(Path("params.yaml").read_text())
    run_cfg = params["runs"][args.run]
    mp = params["models"][run_cfg["model"]]
    ap = params["activations"]
    op = params["output"]

    samples_path = f"data/processed/samples_{run_cfg['family']}.jsonl"
    samples = load_jsonl(samples_path)
    cache_dir = Path(op["activations_dir"]) / args.run
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        print(f"No samples found — skipping extraction → {cache_dir}")
        return

    extract_hidden_states(
        samples=samples,
        model_name=mp["name"],
        cache_dir=cache_dir,
        token_positions=ap["token_positions"],
        mask_positions=ap.get("mask_positions", []),
        layers=ap["layers"],
        batch_size=mp["batch_size"],
        device=mp["device"],
        dtype=mp.get("dtype", "auto"),
    )
    print(f"Activations cached for {len(samples)} samples → {cache_dir}")


if __name__ == "__main__":
    main()
