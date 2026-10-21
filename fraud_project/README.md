# Financial Fraud Detection Model with Dashboard

Two files do the whole project:

| File | What it is |
|---|---|
| `Fraud_Detection_Project.ipynb` | Full pipeline: load 4 datasets -> clean -> ETL to SQLite -> EDA graphs -> feature engineering -> supervised + unsupervised models -> gradient descent -> graph analysis -> streaming simulation + email alert -> adaptive learning -> saves everything the dashboard needs into `artifacts/` |
| `app.py` | Streamlit dashboard (Overview, EDA explorer, Fraud heatmaps, Model performance, Live monitor, Check a transaction) |

## Run it (3 steps)

```bash
pip install -r requirements.txt
```
1. Put the 4 Excel files in the `data/` folder (`credit_card_fraud_dataset.xlsx`, `financial_fraud_detection_dataset.xlsx`,
   `synthetic_fraud_dataset1.xlsx`, `Fraud_Detection.xlsx`).
2. Open the notebook in Jupyter / VS Code and **Run All** (about 1-2 minutes). This creates the `artifacts/` folder.
3. `streamlit run app.py`

> Run the notebook on *your* machine once (step 2) so the saved model matches your scikit-learn version.

## Honest notes (good for viva)
* `bank` (Fraud_Detection.xlsx) is the primary dataset: only 68 frauds, but a very learnable pattern (balance mismatch, TRANSFER / CASH_OUT).
* `credit_card` and `synthetic` have **random labels** (every model gets ROC-AUC ~ 0.5) - no algorithm can learn them; the notebook shows this with a benchmark and statistical tests.
* Email alerts are a **dry run** by default. To send real mail set `ALERT_SENDER`, `ALERT_RECIPIENT`, `ALERT_APP_PASSWORD` (Gmail App Password) - never hard-code a password.
* Kafka / PySpark code from the project brief is included (typos fixed) as reference in notebook section 9.4; the notebook simulates the stream so it runs on any laptop.
