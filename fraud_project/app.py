"""Financial Fraud Detection Dashboard (Streamlit)

Run:
    streamlit run app.py

Before the first run, execute the notebook `Fraud_Detection_Project.ipynb` once (Run All).
It creates the ./artifacts folder (database, model, metrics) that this dashboard reads.

Needs: streamlit, pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
"""
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DB_PATH = ARTIFACTS / "fraud_detection.db"
PALETTE = {0: "#4C78A8", 1: "#E4572E"}                   # legit = blue, fraud = red
LABELS = {0: "Legit", 1: "Fraud"}
LABEL_PALETTE = {"Legit": PALETTE[0], "Fraud": PALETTE[1]}
HUE_ORDER = ["Legit", "Fraud"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------------------
# Loading (cached; the file modification time is part of the cache key, so re-running the
# notebook refreshes the dashboard automatically)
# --------------------------------------------------------------------------------------
def artifacts_ready():
    return all((ARTIFACTS / f).exists() for f in
               ["fraud_detection.db", "model.joblib", "anomaly_model.joblib", "meta.json", "metrics.json"])


@st.cache_data(show_spinner=False)
def _read_table(name, stamp):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
    for c in df.columns:
        if "date" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def table(name):
    return _read_table(name, DB_PATH.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _load_json(name, stamp):
    return json.loads((ARTIFACTS / name).read_text())


def load_json(name):
    return _load_json(name, (ARTIFACTS / name).stat().st_mtime)


@st.cache_resource(show_spinner=False)
def _load_joblib(name, stamp):
    return joblib.load(ARTIFACTS / name)


def load_model():
    return _load_joblib("model.joblib", (ARTIFACTS / "model.joblib").stat().st_mtime)


def load_anomaly():
    return _load_joblib("anomaly_model.joblib", (ARTIFACTS / "anomaly_model.joblib").stat().st_mtime)


# --------------------------------------------------------------------------------------
# Feature engineering. This is the SAME function as in the notebook (Section 4), so a new
# transaction is turned into exactly the features the model was trained on.
# --------------------------------------------------------------------------------------
def engineer_features(df, meta):
    out = pd.DataFrame(index=df.index)
    out["amount"] = df["amount"]
    out["log_amount"] = np.log1p(df["amount"])
    for c in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        out[c] = df[c]
    out["err_orig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    out["err_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    out["orig_emptied"] = (df["newbalanceOrig"] == 0).astype(int)
    out["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    out["unusuallogin"] = df["unusuallogin"]
    for t in meta["types"]:
        out[f"type_{t}"] = (df["type"] == t).astype(int)
    out["time_of_day"] = df["time_of_day"].map(meta["tod_map"])
    out["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    out["branch_freq"] = df["branch"].map(pd.Series(meta["branch_freq"])).fillna(0.0)
    return out[meta["features"]]


def anomaly_score(features, meta):
    """Isolation-Forest anomaly score as a 0-100 percentile of the training data (100 = most unusual)."""
    bundle = load_anomaly()
    z = features[meta["features"]].copy()
    z[meta["log_cols"]] = np.arcsinh(z[meta["log_cols"]])
    raw = -bundle["model"].score_samples(bundle["scaler"].transform(z))
    return np.interp(raw, meta["anomaly_quantiles"], np.linspace(0, 100, len(meta["anomaly_quantiles"])))


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def annotate_bars(ax, fmt="{:,.0f}"):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom", fontsize=9)


def signal_label(roc_auc):
    if roc_auc >= 0.9:
        return "Strong"
    if roc_auc >= 0.65:
        return "Moderate"
    return "None (about random)"


def balanced_sample(df, n_legit=3000, seed=42):
    legit, fraud = df[df["is_fraud"] == 0], df[df["is_fraud"] == 1]
    return pd.concat([legit.sample(min(len(legit), n_legit), random_state=seed), fraud])


def numeric_cols(df, id_cols):
    cols = df.select_dtypes("number").columns
    return [c for c in cols if c not in id_cols and c != "is_fraud"]


def category_cols(df, id_cols):
    out = []
    for c in df.columns:
        if c in id_cols or c == "is_fraud" or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 12:
            out.append(c)
    return out


# --------------------------------------------------------------------------------------
# Page 1: Overview
# --------------------------------------------------------------------------------------
def page_overview(meta, metrics):
    st.subheader("Overview")
    tx = table("transactions")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions (4 datasets)", f"{len(tx):,}")
    c2.metric("Fraud cases", f"{int(tx['is_fraud'].sum()):,}")
    c3.metric("Deployed model", metrics["deployed_model"])
    c4.metric("Alert threshold", f"{meta['threshold']:.2f}")

    bench = pd.DataFrame(metrics["benchmark"]).set_index("dataset")
    summary = (tx.groupby("source")["is_fraud"].agg(transactions="size", fraud_cases="sum", fraud_rate="mean")
               .assign(fraud_rate_pct=lambda d: (100 * d["fraud_rate"]).round(2))
               .join(bench[["cv_roc_auc"]]))
    summary["fraud_signal"] = summary["cv_roc_auc"].map(signal_label)
    summary = summary.rename(columns={"cv_roc_auc": "cv_roc_auc (0.5 = random)"}).drop(columns="fraud_rate")
    st.dataframe(summary.round(3).reset_index())

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.barplot(x=summary.index, y=summary["fraud_rate_pct"], color=PALETTE[1], ax=ax)
        annotate_bars(ax, "{:.2f}%")
        ax.set(title="Fraud rate by dataset", xlabel="", ylabel="fraud rate (%)")
        show(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.barplot(x=bench.index, y=bench["cv_roc_auc"], hue=bench.index, palette="viridis", legend=False, ax=ax)
        ax.axhline(0.5, color="red", ls="--", lw=1)
        annotate_bars(ax, "{:.3f}")
        ax.set(title="How predictable is fraud? (cross-validated ROC-AUC)", xlabel="", ylabel="ROC-AUC", ylim=(0.4, 1.05))
        show(fig)

    st.info(
        "**Data finding:** only the bank dataset (`Fraud_Detection.xlsx`) has a strong, learnable fraud pattern, so "
        "the deployed model is trained on it. In the credit-card and synthetic datasets the fraud labels behave like "
        "random noise (ROC-AUC about 0.5), so their charts are for exploration only."
    )


# --------------------------------------------------------------------------------------
# Page 2: EDA explorer
# --------------------------------------------------------------------------------------
def page_eda(meta):
    st.subheader("EDA explorer")
    names = list(meta["datasets"])
    key = st.selectbox("Dataset", names, index=names.index("bank"), format_func=lambda k: meta["datasets"][k]["label"])
    cfg = meta["datasets"][key]
    df = table(cfg["table"])
    amt, ids = cfg["amount"], cfg["id_cols"]
    st.caption(f"{len(df):,} rows, {int(df['is_fraud'].sum()):,} fraud ({100 * df['is_fraud'].mean():.2f}%)")

    chart = st.selectbox("Chart", ["Class balance (count plot)", "Amount distribution (histogram + KDE)",
                                   "Outliers (box plot, Z-score, IQR)", "Scatter plot", "Correlation heatmap",
                                   "Fraud rate by category"])
    label = df["is_fraud"].map(LABELS)

    if chart.startswith("Class balance"):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=label, hue=label, order=HUE_ORDER, palette=LABEL_PALETTE, legend=False, ax=ax)
        annotate_bars(ax)
        ax.set(title="Legit vs fraud", xlabel="")
        show(fig)

    elif chart.startswith("Amount distribution"):
        use_log = st.checkbox("Log scale (log(1 + amount))", value=bool(df[amt].skew() > 2))
        s = df.sample(min(len(df), 30000), random_state=42)
        plot_df = pd.DataFrame({"value": np.log1p(s[amt]) if use_log else s[amt], "class": s["is_fraud"].map(LABELS)})
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(data=plot_df, x="value", hue="class", hue_order=HUE_ORDER, palette=LABEL_PALETTE, kde=True,
                     stat="density", common_norm=False, element="step", ax=ax)
        ax.set(title="Histogram + KDE by class", xlabel=f"log(1 + {amt})" if use_log else amt)
        show(fig)

    elif chart.startswith("Outliers"):
        use_log = st.checkbox("Log scale (log(1 + amount))", value=bool(df[amt].skew() > 2), key="box_log")
        vals = np.log1p(df[amt]) if use_log else df[amt]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=label, y=vals, hue=label, order=HUE_ORDER, palette=LABEL_PALETTE, legend=False, ax=ax)
        ax.set(title="Box plot of the amount", xlabel="", ylabel=f"log(1 + {amt})" if use_log else amt)
        show(fig)
        rows = {}
        for k, g in df.groupby("is_fraud"):
            s = g[amt]
            z = np.abs((s - s.mean()) / s.std(ddof=0))
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            rows[LABELS[k]] = {"rows": len(s), "|Z| > 3": int((z > 3).sum()),
                               "IQR outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())}
        st.dataframe(pd.DataFrame(rows).T.reset_index().rename(columns={"index": "class"}))
        st.caption("Outliers are not deleted: in fraud detection, unusually large transactions are often the fraud.")

    elif chart == "Scatter plot":
        nums = numeric_cols(df, ids)
        c1, c2 = st.columns(2)
        x = c1.selectbox("X axis", nums, index=nums.index(amt) if amt in nums else 0)
        y_options = [c for c in nums if c != x]
        y = c2.selectbox("Y axis", y_options, index=0)
        use_log = st.checkbox("Log axes (log(1 + value))", value=False)
        s = balanced_sample(df).copy()
        if use_log:
            s[x], s[y] = np.log1p(s[x].clip(lower=0)), np.log1p(s[y].clip(lower=0))
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=s, x=x, y=y, hue=s["is_fraud"].map(LABELS), hue_order=HUE_ORDER, palette=LABEL_PALETTE,
                        alpha=0.6, s=22, ax=ax)
        ax.set_title("Scatter plot (all fraud + a sample of legit)")
        show(fig)

    elif chart == "Correlation heatmap":
        nums = numeric_cols(df, ids) + ["is_fraud"]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[nums].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
        ax.set_title("Correlation heatmap")
        show(fig)

    else:
        cats = category_cols(df, ids)
        default = cats.index(cfg["category"]) if cfg["category"] in cats else 0
        c1, c2 = st.columns(2)
        cat = c1.selectbox("Category column", cats, index=default)
        top_n = c2.slider("Show top N categories (by volume)", 3, 25, 10)
        grouped = df.groupby(df[cat].astype(str))["is_fraud"].agg(["size", "mean"])
        grouped = grouped.sort_values("size", ascending=False).head(top_n).sort_values("mean", ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(grouped) + 1)))
        sns.barplot(x=100 * grouped["mean"], y=grouped.index, color=PALETTE[1], ax=ax)
        ax.set(title=f"Fraud rate (%) by {cat}", xlabel="fraud rate (%)", ylabel="")
        show(fig)
        st.dataframe(grouped.rename(columns={"size": "transactions", "mean": "fraud_rate"}).round(4).reset_index())


# --------------------------------------------------------------------------------------
# Page 3: Fraud heatmaps (unified table)
# --------------------------------------------------------------------------------------
def page_heatmaps():
    st.subheader("Fraud heatmaps")
    tx = table("transactions")
    c1, c2, c3 = st.columns(3)
    source = c1.selectbox("Dataset", ["all"] + sorted(tx["source"].unique()), index=0)
    dims = ["txn_type", "location", "device", "category", "weekday", "month"]
    row_dim = c2.selectbox("Rows", dims, index=dims.index("location"))
    col_dim = c3.selectbox("Columns", dims, index=dims.index("weekday"))
    c4, c5 = st.columns(2)
    metric = c4.radio("Cell value", ["Fraud rate (%)", "Fraud cases", "Transactions"], horizontal=True)
    top_n = c5.slider("Max categories per axis", 4, 25, 12)

    data = tx if source == "all" else tx[tx["source"] == source]
    if row_dim == col_dim:
        st.warning("Choose two different dimensions for rows and columns.")
        return

    def top_values(dim):
        if dim == "weekday":
            return [d for d in WEEKDAYS if d in set(data[dim])]
        return data[dim].astype(str).value_counts().head(top_n).index.tolist()

    rows_keep, cols_keep = top_values(row_dim), top_values(col_dim)
    sub = data.assign(**{row_dim: data[row_dim].astype(str), col_dim: data[col_dim].astype(str)})
    sub = sub[sub[row_dim].isin(rows_keep) & sub[col_dim].isin(cols_keep)]
    if sub.empty or len(rows_keep) < 2 or len(cols_keep) < 2:
        st.warning("Not enough distinct values in this selection (for example, only 'Unknown'). Pick other dimensions or another dataset.")
        return

    agg = {"Fraud rate (%)": "mean", "Fraud cases": "sum", "Transactions": "size"}[metric]
    pivot = sub.pivot_table(index=row_dim, columns=col_dim, values="is_fraud", aggfunc=agg)
    pivot = pivot.reindex(index=rows_keep, columns=cols_keep)
    if metric.startswith("Fraud rate"):
        pivot = pivot * 100
    fig, ax = plt.subplots(figsize=(min(14, 2 + 0.9 * len(cols_keep)), min(10, 2 + 0.45 * len(rows_keep))))
    sns.heatmap(pivot, annot=pivot.size <= 150, fmt=".1f" if metric.startswith("Fraud rate") else ".0f",
                cmap="Reds", linewidths=0.4, ax=ax)
    ax.set_title(f"{metric}: {row_dim} x {col_dim}  ({source})")
    show(fig)
    st.caption("Cells based on few fraud cases are noisy: treat patterns as hints to investigate, not conclusions.")

    st.markdown("**Fraud cases by dataset and transaction type**")
    bar = tx.pivot_table(index="txn_type", columns="source", values="is_fraud", aggfunc="sum", fill_value=0)
    st.bar_chart(bar[bar.sum(axis=1) > 0])


# --------------------------------------------------------------------------------------
# Page 4: Model performance
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _scored_test(_model, stamp, features):
    pf = table("primary_features")
    test = pf[pf["split"] == "test"].reset_index(drop=True)
    return test.assign(score=_model.predict_proba(test[list(features)])[:, 1])


def scored_test(model, meta):
    return _scored_test(model, (ARTIFACTS / "model.joblib").stat().st_mtime, tuple(meta["features"]))


def page_models(meta, metrics, model):
    st.subheader("Model performance (bank dataset)")
    sp = metrics["split"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deployed model", metrics["deployed_model"])
    c2.metric("Train rows", f"{sp['train_rows']:,}")
    c3.metric("Test rows", f"{sp['test_rows']:,}")
    c4.metric("Frauds in test set", sp["test_frauds"])

    sup = pd.DataFrame(metrics["supervised"]).set_index("model")
    st.markdown("**Supervised models** (`cv_*` = 5-fold cross-validation on training data, `test_*` = hold-out test)")
    st.dataframe(sup.round(3).reset_index())
    st.caption("With only a handful of frauds in the test set, differences between the top models are not "
               "statistically meaningful. Accuracy is not shown on purpose: always predicting 'legit' already scores 99%.")
    st.markdown("**Unsupervised anomaly detectors** (trained without fraud labels)")
    st.dataframe(pd.DataFrame(metrics["unsupervised"]).round(3))

    curves = metrics["curves"]
    chosen = st.multiselect("Curves to show", list(curves), default=list(sup.index))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in chosen:
        ax[0].plot(curves[name]["fpr"], curves[name]["tpr"], label=name)
        ax[1].plot(curves[name]["recall"], curves[name]["precision"], label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="Precision-Recall curve (test set)", xlabel="Recall", ylabel="Precision")
    if chosen:
        ax[0].legend(); ax[1].legend()
    show(fig)

    st.markdown("**Try a different alert threshold** (scores come from the deployed model on the hold-out test set)")
    test = scored_test(model, meta)
    thr = st.slider("Alert threshold", 0.01, 0.99, float(meta["threshold"]), 0.01)
    pred = (test["score"] >= thr).astype(int)
    cm = confusion_matrix(test["is_fraud"], pred, labels=[0, 1])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision_score(test['is_fraud'], pred, zero_division=0):.2f}")
    m2.metric("Recall", f"{recall_score(test['is_fraud'], pred):.2f}")
    m3.metric("F1", f"{f1_score(test['is_fraud'], pred, zero_division=0):.2f}")
    m4.metric("False alarms", int(cm[0, 1]))
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred legit", "Pred fraud"], yticklabels=["Actual legit", "Actual fraud"])
        ax.set_title(f"Confusion matrix at threshold {thr:.2f}")
        show(fig)
    with right:
        imp = pd.Series(metrics["feature_importance"]).sort_values(ascending=False).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        imp.plot.barh(color=PALETTE[1], ax=ax)
        ax.set(title="Top features (permutation importance)", xlabel="drop in PR-AUC when shuffled")
        show(fig)
    st.caption("Lower threshold = catch more fraud but raise more false alarms; higher threshold = fewer alarms but more missed fraud.")


# --------------------------------------------------------------------------------------
# Page 5: Live monitor (micro-batch simulation of the Kafka / Spark Streaming flow)
# --------------------------------------------------------------------------------------
ALERT_COLS = ["txn_id", "date", "type", "branch", "amount", "score", "anomaly_pct", "actual (demo only)"]


def render_sim(state, final=False):
    h, alerts = state["history"], state["alerts"]
    real, caught = int(h["real_frauds"].sum()), int(h["frauds_caught"].sum())
    m = st.columns(6)
    m[0].metric("Processed", f"{int(h['txns'].sum()):,}")
    m[1].metric("Alerts raised", int(h["alerts"].sum()))
    m[2].metric("Real frauds so far", real)
    m[3].metric("Caught", caught)
    m[4].metric("Missed", real - caught)
    m[5].metric("False alarms", int(h["alerts"].sum()) - caught)
    left, right = st.columns(2)
    with left:
        st.markdown("**Alerts vs real frauds per batch**")
        st.bar_chart(h.set_index("batch")[["alerts", "real_frauds"]])
    with right:
        st.markdown("**Average anomaly percentile per batch** (Isolation Forest, 100 = most unusual)")
        st.line_chart(h.set_index("batch")[["mean_anomaly_pct"]])
    st.markdown("**Alerts** (highest risk first)")
    if alerts.empty:
        st.write("No alerts yet.")
    else:
        st.dataframe(alerts.sort_values("score", ascending=False)[ALERT_COLS].round({"score": 3, "anomaly_pct": 1, "amount": 2}),
                     hide_index=True)
    if final and not alerts.empty:
        st.download_button("Download alerts (CSV)", alerts[ALERT_COLS].to_csv(index=False).encode(), "fraud_alerts.csv", "text/csv")


def page_live(meta, metrics, model):
    st.subheader("Live monitor (simulated stream)")
    st.caption("Hold-out test transactions arrive in time order, in micro-batches. Each batch is scored by the deployed "
               "model; anything above the threshold raises an alert. This mimics the Kafka / Spark Streaming flow "
               "from the project plan without needing any servers.")
    c1, c2, c3 = st.columns(3)
    batch_size = c1.slider("Batch size", 50, 500, 250, 25)
    thr = c2.slider("Alert threshold", 0.01, 0.99, float(meta["threshold"]), 0.01)
    delay = c3.slider("Pause between batches (seconds)", 0.0, 1.0, 0.25, 0.05)

    test = table("primary_features")
    test = test[test["split"] == "test"].sort_values("date").reset_index(drop=True)
    feats = test[meta["features"]]
    test = test.assign(score=model.predict_proba(feats)[:, 1], anomaly_pct=anomaly_score(feats, meta))
    test["actual (demo only)"] = test["is_fraud"].map(LABELS)

    holder = st.empty()
    if st.button("Start simulation", type="primary"):
        history, frames = [], []
        for b, start in enumerate(range(0, len(test), batch_size)):
            batch = test.iloc[start:start + batch_size]
            flagged = batch[batch["score"] >= thr]
            frames.append(flagged)
            history.append({"batch": b, "txns": len(batch), "alerts": len(flagged), "real_frauds": int(batch["is_fraud"].sum()),
                            "frauds_caught": int(flagged["is_fraud"].sum()), "mean_anomaly_pct": float(batch["anomaly_pct"].mean())})
            state = {"history": pd.DataFrame(history), "alerts": pd.concat(frames, ignore_index=True)}
            with holder.container():
                render_sim(state)
            time.sleep(delay)
        st.session_state["sim"] = state
        with holder.container():
            render_sim(state, final=True)
    elif "sim" in st.session_state:
        with holder.container():
            render_sim(st.session_state["sim"], final=True)
    else:
        holder.info("Press **Start simulation** to stream the test transactions.")


# --------------------------------------------------------------------------------------
# Page 6: Check one transaction
# --------------------------------------------------------------------------------------
EXAMPLES = {
    "Everyday payment": dict(type="PAYMENT", amount=1500.0, old_o=25000.0, new_o=23500.0, old_d=0.0, new_d=0.0,
                             login=11, tod="Afternoon", weekend=False),
    "Account-draining transfer": dict(type="TRANSFER", amount=250000.0, old_o=250000.0, new_o=0.0, old_d=0.0, new_d=0.0,
                                      login=1, tod="Night", weekend=True),
    "Custom": dict(type="CASH_OUT", amount=10000.0, old_o=50000.0, new_o=40000.0, old_d=0.0, new_d=10000.0,
                   login=8, tod="Morning", weekend=False),
}


def page_check(meta, metrics, model):
    st.subheader("Check a transaction")
    st.caption("Enter the details of one transaction and the deployed model returns a fraud risk score.")
    ex = st.selectbox("Start from an example", list(EXAMPLES))
    d = EXAMPLES[ex]
    branches = sorted(meta["branch_freq"], key=meta["branch_freq"].get, reverse=True)[:40]
    types = meta["types"]

    with st.form("check_form"):
        c1, c2, c3 = st.columns(3)
        ttype = c1.selectbox("Transaction type", types, index=types.index(d["type"]), key=f"type_{ex}")
        amount = c2.number_input("Amount", min_value=0.0, value=d["amount"], step=100.0, key=f"amt_{ex}")
        branch = c3.selectbox("Branch (country)", branches, key=f"branch_{ex}")
        c4, c5, c6, c7 = st.columns(4)
        old_o = c4.number_input("Sender balance before", min_value=0.0, value=d["old_o"], step=100.0, key=f"oo_{ex}")
        new_o = c5.number_input("Sender balance after", min_value=0.0, value=d["new_o"], step=100.0, key=f"no_{ex}")
        old_d = c6.number_input("Receiver balance before", min_value=0.0, value=d["old_d"], step=100.0, key=f"od_{ex}")
        new_d = c7.number_input("Receiver balance after", min_value=0.0, value=d["new_d"], step=100.0, key=f"nd_{ex}")
        c8, c9, c10 = st.columns(3)
        login = c8.slider("Unusual-login count", 0, 20, d["login"], key=f"login_{ex}")
        tods = list(meta["tod_map"])
        tod = c9.selectbox("Time of day", tods, index=tods.index(d["tod"]), key=f"tod_{ex}")
        weekend = c10.checkbox("Weekend", value=d["weekend"], key=f"we_{ex}")
        submitted = st.form_submit_button("Score this transaction", type="primary")

    if not submitted:
        return
    row = pd.DataFrame([{"type": ttype, "amount": amount, "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
                         "oldbalanceDest": old_d, "newbalanceDest": new_d, "unusuallogin": login, "time_of_day": tod,
                         "branch": branch, "date": pd.Timestamp("2018-01-06" if weekend else "2018-01-03")}])
    feats = engineer_features(row, meta)
    score = float(model.predict_proba(feats)[:, 1][0])
    anomaly = float(anomaly_score(feats, meta)[0])
    thr = float(meta["threshold"])

    level = "HIGH" if score >= thr else "MEDIUM" if score >= thr / 2 else "LOW"
    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud risk score", f"{score:.3f}")
    m2.metric("Risk level", level)
    m3.metric("Anomaly percentile", f"{anomaly:.0f} / 100")
    st.progress(min(max(score, 0.0), 1.0))
    if level == "HIGH":
        st.error(f"This transaction would raise an alert (score {score:.3f} is at or above the threshold {thr:.2f}).")
    elif level == "MEDIUM":
        st.warning("Borderline: below the alert threshold, but worth a look.")
    else:
        st.success("Looks normal.")

    flags = []
    if ttype in ("TRANSFER", "CASH_OUT"):
        flags.append("Type is TRANSFER or CASH_OUT, the only types that contain fraud in the training data.")
    if new_o == 0 and old_o > 0:
        flags.append("The sender's balance was emptied to exactly 0.")
    if login <= 3:
        flags.append("Very few unusual-login events (fraud cases in this data had fewer than legit ones).")
    if abs(old_o - amount - new_o) > 1:
        flags.append("The sender's balance change does not match the amount sent.")
    st.markdown("**Simple risk indicators** (rules of thumb, not the model's exact reasoning)")
    st.write("\n".join(f"- {f}" for f in flags) if flags else "- None of the usual red flags are present.")
    with st.expander("Features sent to the model"):
        st.dataframe(feats.T.rename(columns={0: "value"}))


# --------------------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------------------
PAGES = ["Overview", "EDA explorer", "Fraud heatmaps", "Model performance", "Live monitor", "Check a transaction"]


def main():
    st.title("Financial Fraud Detection Dashboard")
    if not artifacts_ready():
        st.error("The model files were not found. Open **Fraud_Detection_Project.ipynb**, run all cells once "
                 "(it creates the `artifacts` folder next to this file), then reload this page.")
        st.stop()

    meta, metrics, model = load_json("meta.json"), load_json("metrics.json"), load_model()
    page = st.sidebar.radio("Go to", PAGES)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model: {metrics['deployed_model']}  \nThreshold: {meta['threshold']:.2f}")

    if page == "Overview":
        page_overview(meta, metrics)
    elif page == "EDA explorer":
        page_eda(meta)
    elif page == "Fraud heatmaps":
        page_heatmaps()
    elif page == "Model performance":
        page_models(meta, metrics, model)
    elif page == "Live monitor":
        page_live(meta, metrics, model)
    else:
        page_check(meta, metrics, model)


if __name__ == "__main__":
    main()
