import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import time
from dataclasses import asdict
from .compounds import preprocess
from .targets import rank_targets
from .interfaces import MockGNN, EnsembleUncertainty

def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    p = argparse.ArgumentParser(description="Offline pilot benchmark; MOCK predictions only")
    p.add_argument("command", choices=["benchmark"])
    p.add_argument("--compounds", type=Path, default=Path("data/examples/compounds.csv"))
    p.add_argument("--targets", type=Path, default=Path("data/examples/targets.csv"))
    p.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    p.add_argument("--output", type=Path, default=Path("outputs/pilot"))
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config["backend"] != "mock" or config["device"] != "cpu":
        p.error("Pilot implements only mock backend on CPU")
    start = time.perf_counter()
    accepted, rejected = preprocess(args.compounds, config["max_compounds"])
    if not accepted:
        p.error("No valid compounds")
    targets = rank_targets(args.targets)
    predictions = [asdict(x) for x in MockGNN().predict(accepted)]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "compounds.csv", accepted, ["compound_id", "smiles", "mw"])
    write_csv(out / "rejected.csv", rejected, ["compound_id", "smiles", "reason"])
    (out / "targets.json").write_text(json.dumps(targets, indent=2)+"\n")
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2)+"\n")
    report = dict(mode="MOCK_SMOKE_TEST_NOT_SCIENTIFIC_VALIDATION", config=config,
                  accepted=len(accepted), rejected=len(rejected), targets=len(targets),
                  uncertainty_example=EnsembleUncertainty().summarize([0.,0.,0.]),
                  elapsed_seconds=time.perf_counter()-start,
                  python=platform.python_version(), platform=platform.platform(),
                  versions={k: importlib.metadata.version(k) for k in ["rdkit","numpy"]},
                  input_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest()
                                for x in [args.compounds,args.targets,args.config]})
    (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
