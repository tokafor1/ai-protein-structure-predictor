# AI-Powered Protein Structure Predictor

A beginner-friendly machine learning project that predicts a protein's likely structural class from its amino-acid sequence.

This project uses:
- **Biopython** for protein sequence handling
- **scikit-learn** for training and evaluation
- **Python** for feature extraction, modeling, and prediction

> Note: This is an educational project. It does **not** replace advanced protein folding tools like AlphaFold. It predicts broad structure classes from sequence-based features.

## Project Goal

Given a protein sequence, the model predicts one of three simplified structure classes:

- `alpha` — mostly alpha-helical
- `beta` — mostly beta-sheet
- `mixed` — alpha/beta mixed structure

## Folder Structure

```text
ai_protein_structure_predictor/
├── data/
│   └── sample_proteins.csv
├── models/
├── src/
│   ├── features.py
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train_model.py
```

This trains a Random Forest model and saves it to:

```text
models/protein_structure_model.pkl
```

## Predict a New Protein Sequence

```bash
python src/predict.py "MKTAYIAKQRQISFVKSHFSRQDILDLIC"
```

Example output:

```text
Predicted structure class: alpha
```

## Features Used

The model uses simple sequence-based features, including:

- Sequence length
- Molecular weight
- Aromaticity
- Instability index
- Isoelectric point
- Gravy score
- Amino acid composition
- Alpha-helix favoring amino acid percentage
- Beta-sheet favoring amino acid percentage

## Resume-Friendly Description

AI-Powered Protein Structure Predictor  
Developed a Python-based machine learning project using Biopython and scikit-learn to classify protein sequences into simplified structural categories. Extracted biochemical sequence features and trained a Random Forest classifier to predict protein structure patterns from amino acid sequences.
