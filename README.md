# FaceTTD: Time-to-Death Prediction from Facial Time Series

**FaceTTD** explores whether longitudinal facial images can predict **time-to-death (TTD)** using simple, reproducible pipelines. The repo contains **in-distribution (ID)** experiments on IMDB-FaceTTD and **out-of-distribution (OOD)** evaluation on Wiki-FaceTTD.

---

## Repository Structure

```
facettd/
├── intrinsic_capacity.ipynb                # IMDB-FaceTTD: data prep + ID modeling
├── best_clean_icfaceage_notebook.ipynb     # IMDB-FaceTTD: alt/clean pipeline + analyses
├── 00_wiki_extract.ipynb                   # Wiki-FaceTTD: extract metadata + raw dataset assembly
├── 01_wiki_preprocess.ipynb                # Wiki-FaceTTD: clean and preprocess images/labels
└── 02_wiki_test.ipynb                      # Wiki-FaceTTD: OOD evaluation using IMDB-trained models
```

---

## What’s in Each Notebook?

### IMDB-FaceTTD (In-Distribution)
- **`intrinsic_capacity.ipynb`**  
  Prepares IMDB-FaceTTD dataset (computes TTD, preprocesses portraits, builds train/test splits).  
  Trains XGBoost and Random Forest regressors with subject-level leakage prevention.  

- **`best_clean_icfaceage_notebook.ipynb`**  
  A cleaned, reproducible variant of the IMDB pipeline.  
  Adds stratified group splits by TTD, multiple conditions for longitudinal coverage, and reproducibility checks.  

### Wiki-FaceTTD (Out-of-Distribution)
- **`00_wiki_extract.ipynb`**  
  Extracts metadata from `wiki.mat` / Wikipedia dataset and computes **TTD = deathYear – photo_taken**.  

- **`01_wiki_preprocess.ipynb`**  
  Cleans the extracted Wiki dataset:  
  - Converts portraits to grayscale and resizes them to **64×64**.  
  - Standardizes metadata fields (e.g., age at photo).  
  - Filters out unusable entries (e.g., missing death years).  
  - Outputs aligned feature arrays for OOD testing.  

- **`02_wiki_test.ipynb`**  
  Loads IMDB-trained models and applies them directly to the preprocessed Wiki portraits.  
  Ensures feature dimensionality matches training (handles missing fields like gender/cause-of-death with zero padding).  
  Reports **R²** and **MAE**, including stratified OOD results by TTD horizon (≤60, ≤45, 5–45).  

---

## Quick Start

1. **Run IMDB pipeline (ID)**  
   - Start with `intrinsic_capacity.ipynb` (or `best_clean_icfaceage_notebook.ipynb`) to generate IMDB models.  

2. **Build Wiki pipeline (OOD)**  
   - `00_wiki_extract.ipynb`: Extract Wiki metadata.  
   - `01_wiki_preprocess.ipynb`: Preprocess and align Wiki portraits.  
   - `02_wiki_test.ipynb`: Evaluate IMDB-trained models on Wiki data.  

---

## Environment

Tested with Python 3.10+. Install the essentials:

```bash
pip install -U numpy pandas scikit-learn xgboost pillow matplotlib
```

---

## Reproducibility

- **Split unit**: subject-level to avoid identity leakage  
- **Random seeds**: fixed `random_state=42` for splitting and models  
- **Preprocessing**: grayscale resize (64×64), scaling fit on train only  
- **Models**: XGBoost and Random Forest regressors  
- **Metrics**: R² and MAE (per split; visualized with scatter plots)

---

## Evaluation Protocol

- **In-Distribution (ID)**: Held-out IMDB test subjects (subject-level split).  
- **Out-of-Distribution (OOD)**: Wiki portraits; no refitting (strict generalization test).  
- **TTD Stratification (OOD)**: ≤60, ≤45, 5–45 horizons.  
- **Longitudinal Coverage (ID)**: Compare single vs multi-timepoint subjects.

---

## Citing

If you use FaceTTD or build on these notebooks, please cite:

```bibtex
@misc{facettd2025,
  title        = {FaceTTD: Time-to-Death Inference from Facial Time Series for Mortality Risk Profiling and Interventions,
  author       = {Anonymous Authors},
  year         = {2025},
  howpublished = {NeurIPS TS4H (in review)}

}
```

And baseline models:

```bibtex
@inproceedings{chen2016xgboost,
  title={XGBoost: A Scalable Tree Boosting System},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={KDD},
  year={2016}
}

@article{breiman2001random,
  title={Random Forests},
  author={Breiman, Leo},
  journal={Machine Learning},
  year={2001}
}
```

---

## License

Specify your license (e.g., MIT, Apache-2.0). If unspecified, GitHub defaults to “no license” (others cannot reuse).

---

## Acknowledgments

This work builds on open-source tooling (`numpy`, `pandas`, `scikit-learn`, `xgboost`, `Pillow`, `matplotlib`) and public portrait datasets (IMDB, Wikipedia). Please respect the dataset providers’ terms.
