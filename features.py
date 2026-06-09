"""
Feature extraction utilities for protein sequences.
"""

from typing import Dict
import re
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

ALPHA_FAVORING = set("AELMKQRH")
BETA_FAVORING = set("VIYFWTC")


def clean_sequence(sequence: str) -> str:
    """
    Remove invalid characters and uppercase the sequence.
    """
    if not isinstance(sequence, str):
        raise ValueError("Sequence must be a string.")

    sequence = sequence.upper().strip()
    sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", sequence)

    if len(sequence) == 0:
        raise ValueError("Sequence does not contain valid amino acid characters.")

    return sequence


def extract_features(sequence: str) -> Dict[str, float]:
    """
    Convert a protein sequence into numerical features.
    """
    sequence = clean_sequence(sequence)
    analysis = ProteinAnalysis(sequence)

    features = {
        "length": len(sequence),
        "molecular_weight": analysis.molecular_weight(),
        "aromaticity": analysis.aromaticity(),
        "instability_index": analysis.instability_index(),
        "isoelectric_point": analysis.isoelectric_point(),
        "gravy": analysis.gravy(),
    }

    # Amino acid composition
    aa_percent = analysis.get_amino_acids_percent()
    for aa in AMINO_ACIDS:
        features[f"pct_{aa}"] = aa_percent.get(aa, 0.0)

    # Simple structural tendency features
    alpha_count = sum(1 for aa in sequence if aa in ALPHA_FAVORING)
    beta_count = sum(1 for aa in sequence if aa in BETA_FAVORING)

    features["alpha_favoring_pct"] = alpha_count / len(sequence)
    features["beta_favoring_pct"] = beta_count / len(sequence)

    return features


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature table from a dataframe containing a 'sequence' column.
    """
    feature_rows = [extract_features(seq) for seq in df["sequence"]]
    return pd.DataFrame(feature_rows)
