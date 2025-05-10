# 🔍 ARDIAN Capstone – Similar Company Search Prototype

This repository contains a **lightweight prototype** of a larger industrial project implemented on the **ARdian Azure server**. It serves as a **local testing environment** for an interactive search tool that recommends similar companies based on textual business descriptions and financial metadata.

The core objective is to **match companies** using semantic embeddings, dimension reduction, and user-specified filters (e.g., country, sector, FTE bounds). This small-scale version allows for rapid experimentation and is modular enough to evolve further.

---

## 📁 Repository Structure
```
vincentg1234-ardian_capstone.git/
├── README.md               ← This file
├── NB_main.ipynb           ← Interactive notebook for local testing
├── requirements.txt        ← List of dependencies
├── data/                   ← Sample dataset for local usage
└── Scripts/                ← Core logic (modular Python scripts)
    ├── main.py                         ← Entry point to run full pipeline
    ├── filter_user/
    │   ├── ask_user.py                 ← User prompts and CLI interactions
    │   └── filter.py                   ← Data filtering and description preprocessing
    └── language_model_folder/
        ├── language_model.py           ← Embedding generation and similarity ranking
        └── PCA_functions.py            ← Dimensionality reduction using PCA
```

## ⚙️ Setup Instructions

Python version: Make sure you’re using Python 3.11

Install dependencies:

bash
```
pip install -r requirements.txt
```
It’s strongly recommended to use a virtual environment:

bash
```
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

Run the notebook:
Open NB_main.ipynb to run the search pipeline step-by-step with a small dataset.

## 🧠 What the Pipeline Does

Filters companies based on user input (country, sector, size)

Uses Sentence-BERT to embed business descriptions

Enriches descriptions with user-provided or inferred keywords

Applies PCA to reduce vector dimensions while preserving semantic structure

Ranks and displays the top-N most similar companies

You’ll be prompted interactively in the terminal or notebook to guide each step.

## 📌 Notes
The dataset in the data/ folder is a minimal subset for local prototyping. The full version runs on Ardian's secure Azure infrastructure.

Please create a new branch before making any changes.

Be careful with git add and commits: avoid pushing unwanted cache or system files.

## 📧 Contact
This repo is maintained as part of a capstone project at ARDIAN. For any questions or suggestions, feel free to reach out to the team.
