# Improving Dynamic ESG: Robust & Multilingual ESG Signal Detection from News Articles
## GitHub Repository Link
https://github.com/OrienPaxx/improving-dynamic-esg
**Course:** Artificial Intelligence — Final Project (Week 17 Documentation & Submission)  
**Student:** Lipika Basumatary (1127148)  
**Teammate:** Cissy (1127192)

## 1) Project Summary
ESG (Environmental, Social, Governance) ratings are typically updated annually, but ESG-related events occur daily in news.  
This project builds a **text classification pipeline** to detect ESG signals from news articles using the **ESG-FTSE corpus** and compares a strong traditional baseline (TF‑IDF + Logistic Regression) against transformer models (BERT/DistilBERT/XLM‑RoBERTa).

Key focus areas:
- Data preprocessing and label encoding (E/S/G)
- Model comparison (baseline vs. transformers)
- Handling class imbalance (class weights)
- Robustness experiments (data augmentation / adversarial-style noise)

## 2) Repository Structure
```
.
├── notebooks/
│   └── ESG_ratings_Final_Project.ipynb
├── data/                       # NOT committed (see .gitignore)
│   └── esg_ftse_corpus.json     # place dataset here
├── results/
│   ├── xlm_roberta_tuned_predictions.csv
│   └── figures/                # plots (optional)
├── docs/
│   └── one_page_instructions.pdf
├── requirements.txt
└── README.md
```

## 3) Environment Setup
### Option A — Google Colab (recommended)
1. Open `notebooks/ESG_ratings_Final_Project.ipynb` in Colab.
2. Follow the notebook top-to-bottom.

### Option B — Local (Python 3.10+)
```bash
git clone <YOUR_GITHUB_REPO_LINK>
cd <REPO_NAME>
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Dataset
This repo expects the ESG‑FTSE JSON file:
- `data/esg_ftse_corpus.json`

If you received it from the course / instructor, download it and place it in `data/` (the folder is gitignored).  
If your course requires sharing a link instead of the file itself, add the official dataset link here:

- Dataset link: - Dataset link: Provided via course materials (ESG-FTSE corpus)

## 5) How to Reproduce Results (Notebook)
1. Ensure the dataset is available as `data/esg_ftse_corpus.json`.
2. Run the notebook **in order**.
3. Expected outputs:
   - Printed evaluation metrics (Accuracy, Macro‑F1, class-wise report)
   - `results/xlm_roberta_tuned_predictions.csv` (generated)

## 6) Training Your Own Model (Recommended Settings)
In the notebook, the tuned XLM‑RoBERTa setup uses:
- `num_train_epochs = 5`
- `learning_rate = 5e-5`
- `weight_decay = 0.001`

## 7) Results Summary (from final presentation)
- Best model: **XLM‑RoBERTa (tuned)**
- Test Accuracy: **0.78**
- Macro F1: **0.746**

## 8) Notes / Limitations (Optional)
- Results can vary slightly due to randomness (set seed for closer reproducibility).
- Training transformer models may require GPU (Colab T4 is sufficient).
- Cross-lingual extension is discussed in the presentation, but this notebook focuses primarily on ESG‑FTSE (English) unless additional data is provided.

---
If anything fails to run, please open an issue in the repository with the error log and your environment details.
