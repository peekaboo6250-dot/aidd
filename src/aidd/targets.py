"""Manual evidence triage: demonstration ratings are not disease evidence."""
import csv

def rank_targets(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        scores = [float(row[k]) for k in ("disease", "ligand_data", "structure")]
        if any(not 0 <= x <= 5 for x in scores):
            raise ValueError("Target ratings must be finite values in [0, 5]")
        row["priority"] = round(sum(w*x for w,x in zip((0.5,0.3,0.2), scores)), 4)
    return sorted(rows, key=lambda r: (-r["priority"], r["target"]))
