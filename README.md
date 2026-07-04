# Parakh — Market-Potential Validation Dashboard

An interactive **Streamlit** dashboard that validates the Parakh business idea (a pre-trade
risk tool for India's retail investors) using machine learning on a cleaned dataset of
**1,000 investors** and **7,876 trades**.

It covers the four graded techniques:

| Page | Technique | Algorithms |
|------|-----------|-----------|
| Classification | Predict whether a trade will lose money | KNN, Decision Tree, Random Forest, Gradient Boosted Trees (+ accuracy, precision, recall, F1, ROC-AUC) |
| Segmentation | Group investors into customer segments | K-Means (elbow + silhouette, PCA plot, named segments) |
| Association Rules | Which habits cause losses | Apriori — support, confidence, lift |
| Regression | Predict willingness-to-pay | Linear, Ridge, Lasso |
| Business Verdict | Launch decision & recommendations | — |

All models train **live** from the CSVs in `data/`, so every number is reproducible.

---

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501).

Prefer the terminal? Run the whole analysis without the dashboard:

```bash
python run_analysis.py
```

---

## Deploy free on Streamlit Community Cloud (via GitHub)

1. **Create a GitHub repo** and upload this whole folder (keep the `data/` folder — the app needs it).
   ```bash
   git init
   git add .
   git commit -m "Parakh market validation dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/parakh-dashboard.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your repo, branch `main`, and set the main file to **`app.py`**.
4. Click **Deploy**. Streamlit Cloud reads `requirements.txt` automatically and gives you a public URL.

That's it — share the link.

---

## Files

```
app.py                 Streamlit dashboard (6 pages)
analysis.py            All ML functions (shared, reproducible)
run_analysis.py        Terminal version of the full analysis
requirements.txt       Python dependencies
.streamlit/config.toml Theme (neutral palette)
data/
  investors.csv        Master 1 — 1,000 investors x 59 cols
  trades.csv           Master 2 — 7,876 trades x 49 cols
BUSINESS_INSIGHTS.md   Written findings, risks & launch verdict
```

## Notes
- Target for classification is `loss_occurred`; only **pre-trade** features are used (no leakage from `loss_amount`, `loss_pct`, regret, etc.).
- Data is **synthetic**, generated for coursework — not real individuals.
