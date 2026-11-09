import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Learning Difficulty Detection", layout="wide")
st.title("Adaptive Explainable AI-Based Student Learning Difficulty Detection")
st.caption("Random Forest model + SHAP explanations")

LEVEL_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove hidden characters, line breaks and extra spaces from column names."""
    frame.columns = [
        re.sub(r"\s+", " ", str(c).replace("\ufeff", "").replace("ï»¿", "")).strip()
        for c in frame.columns
    ]
    return frame


# ---------------------------------------------------------------- data loading
st.sidebar.header("1. Dataset")
uploaded = st.sidebar.file_uploader("Upload your CSV (optional)", type=["csv"])

if uploaded is not None:
    df = clean_columns(pd.read_csv(uploaded, encoding="utf-8-sig"))
    st.sidebar.success("Using uploaded file")
else:
    here = os.path.dirname(os.path.abspath(__file__))
    csvs = sorted(
        glob.glob(os.path.join(here, "*.csv"))
        + glob.glob(os.path.join(here, "data", "*.csv"))
    )
    if not csvs:
        st.warning("No CSV found in this folder or in data/. Upload your dataset in the sidebar.")
        st.stop()
    names = {os.path.relpath(c, here): c for c in csvs}
    chosen = st.sidebar.selectbox("Dataset file", list(names))
    df = clean_columns(pd.read_csv(names[chosen], encoding="utf-8-sig"))

df = df.dropna().drop_duplicates().reset_index(drop=True)  # same as notebook

# target column: guess the one with "difficulty" in the name
cols = list(df.columns)
guess = next((i for i, c in enumerate(cols) if "difficult" in c.lower()), len(cols) - 1)
target = st.sidebar.selectbox("Target column (what we predict)", cols, index=guess)

features = [c for c in cols if c != target]
drop_cols = st.sidebar.multiselect(
    "Columns to ignore (IDs, names, etc.)", features,
    default=[c for c in features if "id" == c.lower() or "name" in c.lower()],
)
features = [c for c in features if c not in drop_cols]


# ---------------------------------------------------------------- preprocessing
@st.cache_resource(show_spinner="Training Random Forest...")
def train(data: pd.DataFrame, feats: tuple, tgt: str):
    X = data[list(feats)].copy()
    y = data[tgt].copy()

    encoders = {}
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            cats = sorted(X[c].astype(str).unique())
            encoders[c] = cats
            X[c] = X[c].astype(str).map({v: i for i, v in enumerate(cats)})

    target_classes = None
    if not pd.api.types.is_numeric_dtype(y):
        target_classes = sorted(y.astype(str).unique())
        y = y.astype(str).map({v: i for i, v in enumerate(target_classes)})

    # Same settings as the project notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True
    )
    # Stable settings: give the same result in every scikit-learn version
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=50,
        max_features=None,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))
    return model, encoders, target_classes, acc, cm, X_train


model, encoders, target_classes, acc, cm, X_train = train(
    df, tuple(features), target
)
class_ids = list(model.classes_)


def class_label(i):
    if target_classes is not None:
        return target_classes[int(i)]
    return LEVEL_NAMES.get(int(i), str(i))


# ---------------------------------------------------------------- tabs
tab1, tab2, tab3 = st.tabs(["Predict a student", "Model performance", "Dataset"])

with tab1:
    st.subheader("Enter student details")
    left, right = st.columns([1, 1])
    values = {}
    with left:
        for c in features:
            if c in encoders:
                values[c] = st.selectbox(c, encoders[c])
            elif pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 6:
                opts = sorted(int(v) for v in df[c].unique())
                values[c] = st.selectbox(c, opts, index=opts.index(int(df[c].median())) if int(df[c].median()) in opts else 0)
            else:
                col = df[c]
                lo, hi = float(col.min()), float(col.max())
                default = float(col.median())
                if pd.api.types.is_integer_dtype(col):
                    values[c] = st.number_input(
                        c, min_value=int(lo), max_value=int(hi), value=int(default), step=1
                    )
                else:
                    values[c] = st.number_input(
                        c, min_value=lo, max_value=hi, value=default
                    )
        go = st.button("Predict learning difficulty", type="primary")

    with right:
        if go:
            row = pd.DataFrame([values])
            for c, cats in encoders.items():
                row[c] = cats.index(str(row[c].iloc[0]))
            row = row[features]

            pred = model.predict(row)[0]
            proba = model.predict_proba(row)[0]

            st.markdown(f"### Predicted difficulty level: **{class_label(pred)}**")
            st.write("Confidence for each level:")
            st.bar_chart(
                pd.Series(proba, index=[class_label(i) for i in class_ids])
            )

            # ---- SHAP explanation (plain-language + chart)
            st.markdown("#### Why this prediction? (SHAP explanation)")
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(row)
            k = class_ids.index(pred)
            if isinstance(sv, list):
                vals = np.array(sv[k])[0]
            else:
                sv = np.array(sv)
                vals = sv[0, :, k] if sv.ndim == 3 else sv[0]

            level = class_label(pred)
            shap_s = pd.Series(vals, index=features)
            total = float(np.abs(shap_s).sum())

            def clean(name):
                return re.sub(r"\s*\(.*?\)", "", name).strip()

            ranked = shap_s.abs().sort_values(ascending=False)
            top = [c for c in ranked.index
                   if total > 0 and abs(shap_s[c]) / total >= 0.01][:5]

            if top:
                names = ", ".join(clean(c) for c in top[:3])
                st.markdown(
                    f"The model predicted **{level}** difficulty mainly because of **{names}**."
                )
                for c in top:
                    v = float(shap_s[c])
                    share = abs(v) / total * 100
                    x = values[c]
                    desc = f"{clean(c)} = {x}"
                    if c not in encoders:
                        avg = float(df[c].mean())
                        span = float(df[c].max() - df[c].min()) or 1.0
                        if abs(float(x) - avg) <= 0.02 * span:
                            cmp_txt = "close to"
                        elif float(x) > avg:
                            cmp_txt = "higher than"
                        else:
                            cmp_txt = "lower than"
                        desc += f" ({cmp_txt} the dataset average of {avg:.1f})"
                    if v > 0:
                        effect = f"pushed the prediction **towards** {level} difficulty"
                    else:
                        effect = f"pushed the prediction **away from** {level} difficulty"
                    st.markdown(
                        f"- **{desc}** {effect} (about {share:.0f}% of the total influence)."
                    )
                if len(top) < len(features):
                    st.caption("The remaining columns had almost no effect on this prediction.")
            else:
                st.write("No single feature had a noticeable effect on this prediction.")

            top8 = shap_s.reindex(ranked.index[:8]).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6, max(2.5, 0.45 * len(top8))))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in top8.values]
            ax.barh([clean(i) for i in top8.index], top8.values, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(f"Effect on predicted level: {level}")
            st.pyplot(fig)
            st.caption(
                "Red bars push the prediction towards this level; "
                "blue bars push it away from it."
            )
        else:
            st.info("Fill in the details on the left and click **Predict**.")

with tab2:
    st.subheader("Model performance")
    c1, c2 = st.columns([1, 2])
    c1.metric("Accuracy (20% test data)", f"{acc * 100:.2f} %")
    c1.write("Model: Random Forest (50 trees, max depth 3, balanced classes)\n\nTrain/Test split: 80% / 20% (stratified)\n\nDuplicate rows removed before training")
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [class_label(i) for i in class_ids]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    c2.pyplot(fig)

    st.markdown("#### Overall feature importance")
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    fig2, ax2 = plt.subplots(figsize=(6, max(2.5, 0.4 * len(features))))
    ax2.barh(imp.index, imp.values, color="#4c78a8")
    st.pyplot(fig2)

with tab3:
    st.subheader("Dataset preview")
    st.write(f"{df.shape[0]} rows x {df.shape[1]} columns")
    st.dataframe(df.head(50))
