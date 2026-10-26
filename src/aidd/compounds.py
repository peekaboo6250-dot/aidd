"""Streaming SMILES cleanup with explicit rejection log; no potency claims."""
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

def preprocess(path, limit=1000):
    if not 1 <= limit <= 10000:
        raise ValueError("Pilot limit must be 1..10000")
    accepted, rejected, seen, ids = [], [], set(), set()
    with open(path, newline="", encoding="utf-8") as f:
        for index, row in enumerate(csv.DictReader(f)):
            if index >= limit:
                raise ValueError("Input exceeds pilot limit; explicitly reduce input")
            ident, smiles = row["compound_id"].strip(), row["smiles"].strip()
            reason = None
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if not ident or ident in ids:
                reason = "missing_or_duplicate_id"
            elif mol is None or mol.GetNumAtoms() == 0:
                reason = "invalid_smiles"
            else:
                mol = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(mol))
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                if canonical in seen:
                    reason = "duplicate_structure"
                elif any(a.GetAtomicNum() not in (1,5,6,7,8,9,14,15,16,17,35,53) for a in mol.GetAtoms()):
                    reason = "unsupported_element"
                elif mol.GetNumHeavyAtoms() > 100:
                    reason = "too_many_atoms"
                else:
                    seen.add(canonical)
                    accepted.append(dict(compound_id=ident, smiles=canonical,
                                         mw=round(Descriptors.MolWt(mol), 4)))
            ids.add(ident)
            if reason:
                rejected.append(dict(compound_id=ident, smiles=smiles, reason=reason))
    return accepted, rejected
