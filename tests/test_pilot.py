import json
from pathlib import Path
import subprocess
import sys
import pytest
from aidd.compounds import preprocess
from aidd.targets import rank_targets
from aidd.interfaces import AIDocking, GNN, EnsembleUncertainty

ROOT = Path(__file__).resolve().parents[1]

def test_cleanup_and_limits():
    path = ROOT / "data/examples/compounds.csv"
    good, bad = preprocess(path)
    assert len(good) == 5 and len(bad) == 3
    assert {x["reason"] for x in bad} == {"duplicate_structure","invalid_smiles"}
    assert next(x for x in good if x["compound_id"] == "salt")["smiles"] == "CCN"
    with pytest.raises(ValueError): preprocess(path, 2)
    with pytest.raises(ValueError): preprocess(path, 0)

def test_targets_and_invalid_rating(tmp_path):
    ranked = rank_targets(ROOT / "data/examples/targets.csv")
    assert [r["target"] for r in ranked] == ["DEMO_A","DEMO_B","DEMO_C"]
    p = tmp_path / "bad.csv"
    p.write_text("target,disease,ligand_data,structure\nX,nan,2,3\n")
    with pytest.raises(ValueError): rank_targets(p)

def test_interfaces_do_not_fake_inference():
    with pytest.raises(NotImplementedError): AIDocking().predict([], "receptor")
    with pytest.raises(NotImplementedError): GNN().predict([])
    u = EnsembleUncertainty()
    assert u.summarize([1.,2.,3.])["sample_std"] == 1.
    assert u.summarize([1.,2.])["calibrated"] is False
    with pytest.raises(ValueError): u.summarize([1.])
    with pytest.raises(ValueError): u.summarize([float("nan"),1.])

def test_cli_repeatable(tmp_path):
    for name in ("one","two"):
        subprocess.run([sys.executable,"-m","aidd.cli","benchmark","--output",str(tmp_path/name)],
                       cwd=ROOT, check=True, capture_output=True)
    for filename in ("compounds.csv","rejected.csv","predictions.json","targets.json"):
        assert (tmp_path/"one"/filename).read_bytes() == (tmp_path/"two"/filename).read_bytes()
    report=json.loads((tmp_path/"one/report.json").read_text())
    assert report["accepted"] == 5 and report["rejected"] == 3
    assert len(report["input_sha256"]) == 3
