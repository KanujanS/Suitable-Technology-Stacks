# A MACHINE LEARNING APPROACH FOR RECOMMENDING SUITABLE TECHNOLOGY STACKS

This repository supports the research project "A MACHINE LEARNING APPROACH FOR RECOMMENDING SUITABLE TECHNOLOGY STACKS BASED ON PROJECT CHARACTERISTICS". It contains the full pipeline used in the study: data collection from GitHub, cleaning and preprocessing, exploratory data analysis (EDA), feature engineering, and model training (multiple machine learning models and ensembles) that predict frontend, backend, and database technologies for a project based on its characteristics.

Repository home: the notebook that collects GitHub projects and the scripts/notebooks used for preprocessing, EDA and model training are included so experiments can be reproduced and extended.

Repository structure (key files)

- Data collection
  - [github_collector.ipynb] — Colab notebook that queries the GitHub Search API and extracts repository metadata, README text, topics and language stats. Default output: `github_projects_data.xlsx`.

- Data cleaning & preprocessing
  - [data_cleaning_final.ipynb] — interactive cleaning steps and checks used during development.
  - [preprocessing_final.py] — preprocessing pipeline that selects ML-relevant columns, groups rare classes into "Other", encodes ordinal/categorical features, builds TF-IDF features from requirement text, and saves artifacts: `tfidf_vectorizer.pkl`, `label_encoder_frontend.pkl`, `label_encoder_backend.pkl`, `label_encoder_database.pkl`, and `train_val_test_splits.pkl`.

- Exploratory Data Analysis (EDA)
  - [eda_final.py] — generates charts (domain distributions, heatmaps of Domain vs Technology, word clouds and other EDA artifacts) and saves PNGs (e.g. `eda_01_domain_counts.png`, `eda_02_domain_vs_frontend.png`).
  - [Extended_Models.ipynb] — notebook with additional model experiments and ablation studies.

- Model training
  - [model_training_final_3.py] — trains multiple classifiers and ensembles used in the paper, including CatBoost, XGBoost, LightGBM, a sentence-transformer + LightGBM pipeline (NoBERT), and stacking ensembles. Saves model artefacts and evaluation outputs to `outputs/`.

Additional files

- `github_projects_cleaned.xlsx` (expected intermediate file produced after cleaning) — used by training scripts.
- Saved artifacts produced by preprocessing/training:
  - `tfidf_vectorizer.pkl` — TF-IDF vocabulary used for feature construction
  - `label_encoder_frontend.pkl`, `label_encoder_backend.pkl`, `label_encoder_database.pkl` — encoders to decode model outputs
  - `train_val_test_splits.pkl` — pre-made X/y train/val/test splits for reproducible training
  - `outputs/` — directory where model checkpoints, metrics, and plots are saved during training

What the pipeline predicts

- Targets: Frontend_Tech, Backend_Tech, Database
- Inputs (representative features): Domain, Functional_Requirements (text), Non_Functional_Requirements (text), Project_Size, Team_Size, Budget_Level, Duration_Months, Deployment, Primary_Language, plus TF-IDF features constructed from FR+NFR and one-hot/ordinal encodings.

High-level pipeline summary

1. Data collection: run `github_collector.ipynb` (in Colab or locally) to produce an initial dataset of GitHub projects and inferred technology labels.
2. Cleaning & labeling: use `data_cleaning_final.ipynb` and `preprocessing_final.py` to clean, group rare classes, encode features, vectorize text (TF-IDF) and produce train/val/test splits.
3. Exploratory analysis: `eda_final.py` creates charts and tables to validate assumptions (e.g. technology prevalence by domain).
4. Model training: `model_training_final_3.py` trains multiple models, evaluates on validation and test sets, and saves results under `outputs/`.

Reproducing the experiments (suggested steps)

1. Clone the repository and change to the project folder:

   cd "<path-to-repo>"

2. (Optional but recommended) Create and activate a virtual environment:

   python3 -m venv venv
   source venv/bin/activate

3. Install required Python packages (packages used across notebooks and scripts):

   pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl
   pip install xgboost lightgbm catboost sentence-transformers wordcloud

   Note: If GPU acceleration is required for heavy transformer usage, follow the sentence-transformers installation notes.

4. Data collection (Colab recommended for convenience):
   - Open [github_collector.ipynb] in Colab or locally.
   - Provide a GitHub personal access token where prompted and run the cells.
   - The notebook outputs `github_projects_data.xlsx`.

5. Cleaning & preprocessing:
   - Open `data_cleaning_final.ipynb` or run `preprocessing_final.py` in an environment that provides the dataset (the Colab notebooks use `google.colab.files.upload()` interaction).
   - Preprocessing produces `github_projects_cleaned.xlsx`, `tfidf_vectorizer.pkl`, label encoder pickles, and `train_val_test_splits.pkl`.

6. Run EDA:
   - Run `python eda_final.py` (or open the EDA notebook) to regenerate EDA charts.

7. Model training:
   - Ensure `train_val_test_splits.pkl` and `github_projects_cleaned.xlsx` are in the same folder as `model_training_final_3.py`.
   - Run:

       python model_training_final_3.py

   - Results (metrics, confusion matrices and model pickles) are saved to `outputs/`.

Requirements (summary)

- Python 3.8+ recommended
- Major Python libraries used in codebase: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, openpyxl, xgboost, lightgbm, catboost, sentence-transformers, wordcloud
- A GitHub personal access token for the data collection notebook

Notes & reproduction tips

- The notebooks were developed and run in Google Colab during the research — some cells use `google.colab.files.upload()` for convenience. When running locally, replace those upload steps with direct file paths.
- Preprocessing uses deterministic groupings and encoding mappings (rare-class grouping into "Other", ordinal maps for Project_Size / Budget_Level / Deployment). For exact reproducibility, use the provided preprocessing script and saved encoders.
- Long-running model variants (e.g., NoBERT / transformer-based features) are optional and controlled via configuration flags in `model_training_final_3.py`.

Contact & citation

If you use this dataset or code in your work, please cite the project and contact the repository owner for questions or collaboration opportunities.

