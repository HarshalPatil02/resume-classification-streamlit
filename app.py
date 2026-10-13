import io
import os
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Resume Classification", page_icon="📄", layout="wide")

st.title("📄 Resume Classification & Role Matching")
st.write(
    "Train a text-classification model from resumes labelled by role, evaluate it, "
    "and predict the most likely role for uploaded resumes or pasted text."
)
st.info(
    "This app follows the uploaded notebook's approach: `TEXT` as the resume text, "
    "`ROLE_FOLDER` as the target label, TF-IDF with English stop-word removal and "
    "up to 3,000 features, and Logistic Regression."
)

@st.cache_data(show_spinner=False)
def read_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


def extract_resume_text(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".docx"):
        from docx import Document
        document = Document(io.BytesIO(raw))
        parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(parts)
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError("Unsupported file type. Upload a .txt, .docx, or text-based .pdf file.")


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=3000)),
        ("model", LogisticRegression(max_iter=1000, class_weight=None)),
    ])


def normalize_dataset(frame):
    required = {"TEXT", "ROLE_FOLDER"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "The dataset is missing required column(s): " + ", ".join(sorted(missing))
            + ". Expected the notebook's `TEXT` and `ROLE_FOLDER` columns."
        )
    clean = frame.dropna(subset=["TEXT", "ROLE_FOLDER"]).copy()
    clean["TEXT"] = clean["TEXT"].astype(str).str.strip()
    clean["ROLE_FOLDER"] = clean["ROLE_FOLDER"].astype(str).str.strip()
    clean = clean[(clean["TEXT"] != "") & (clean["ROLE_FOLDER"] != "")]
    return clean

with st.sidebar:
    st.header("1. Training dataset")
    st.caption("Upload `resume_classification_datasets.csv` created by your notebook.")
    dataset_upload = st.file_uploader("Labeled resume CSV", type=["csv"])
    st.divider()
    st.header("Model settings")
    test_size = st.slider("Test set size (%)", min_value=10, max_value=40, value=20, step=5)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

if dataset_upload is None:
    st.warning("Upload your labeled CSV to train the classifier. It should contain `TEXT` and `ROLE_FOLDER` columns.")
    with st.expander("Expected CSV format"):
        st.dataframe(pd.DataFrame({
            "FILE_NAME": ["resume_001.docx", "resume_002.pdf"],
            "ROLE_FOLDER": ["Data Scientist", "SQL Developer"],
            "TEXT": ["Python, pandas, machine learning...", "SQL, database design, queries..."],
            "LABEL": ["Resume", "Resume"],
        }), use_container_width=True)
    st.stop()

try:
    raw_df = read_csv(dataset_upload.getvalue())
    data = normalize_dataset(raw_df)
except Exception as exc:
    st.error(f"Could not read the training dataset: {exc}")
    st.stop()

if data.empty:
    st.error("No usable rows found. Check that TEXT and ROLE_FOLDER contain non-empty values.")
    st.stop()

class_counts = data["ROLE_FOLDER"].value_counts()
num_classes = data["ROLE_FOLDER"].nunique()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Usable resumes", f"{len(data):,}")
m2.metric("Distinct role labels", f"{num_classes:,}")
m3.metric("Median word count", f"{int(data['TEXT'].str.split().str.len().median()):,}")
m4.metric("Smallest role group", f"{int(class_counts.min()):,}")

with st.expander("Preview dataset and role distribution"):
    st.dataframe(data.head(20), use_container_width=True)
    role_counts = class_counts.rename_axis("Role").reset_index(name="Resume count")
    st.dataframe(role_counts, use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, max(3, min(8, len(role_counts) * 0.35))))
    show_counts = role_counts.sort_values("Resume count", ascending=True).tail(25)
    ax.barh(show_counts["Role"], show_counts["Resume count"])
    ax.set_xlabel("Resume count")
    ax.set_title("Resume count by role (up to 25 largest groups)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Evaluation split: use stratification only if every class can appear in both partitions.
y = data["ROLE_FOLDER"]
X = data["TEXT"]
class_counts = y.value_counts()
can_stratify = len(class_counts) > 1 and class_counts.min() >= 2
stratify_y = y if can_stratify else None

st.header("2. Train and evaluate")
if not can_stratify:
    st.warning("At least one role has fewer than two resumes, so the evaluation split cannot be stratified. Results may be less representative.")

if st.button("Train classifier", type="primary"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=int(random_state), stratify=stratify_y
        )
        evaluation_model = build_pipeline()
        evaluation_model.fit(X_train, y_train)
        y_pred = evaluation_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Refit a production model on all labelled rows for subsequent predictions.
        final_model = build_pipeline()
        final_model.fit(X, y)
        st.session_state["resume_model"] = final_model
        st.session_state["resume_labels"] = sorted(y.unique().tolist())
        st.session_state["resume_accuracy"] = accuracy
        st.session_state["resume_report"] = report
        st.session_state["resume_confusion"] = (y_test.tolist(), y_pred.tolist(), sorted(set(y_test) | set(y_pred)))
        st.session_state["resume_data_rows"] = len(data)
        st.success("Model trained. Evaluation metrics are shown below; predictions use a final model refit on all uploaded labelled data.")
    except Exception as exc:
        st.error(f"Training failed: {exc}")

if "resume_model" in st.session_state:
    accuracy = st.session_state["resume_accuracy"]
    report = st.session_state["resume_report"]
    metric_cols = st.columns(3)
    metric_cols[0].metric("Holdout accuracy", f"{accuracy:.1%}")
    weighted = report.get("weighted avg", {})
    metric_cols[1].metric("Weighted precision", f"{weighted.get('precision', 0):.1%}")
    metric_cols[2].metric("Weighted F1-score", f"{weighted.get('f1-score', 0):.1%}")

    with st.expander("Detailed classification report"):
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

    actual, predicted, labels = st.session_state["resume_confusion"]
    if len(labels) <= 20:
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.55), max(6, len(labels) * 0.45)))
        ConfusionMatrixDisplay.from_predictions(
            actual, predicted, labels=labels, display_labels=labels,
            xticks_rotation=90, cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_title("Holdout confusion matrix")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption("Confusion matrix hidden because there are more than 20 role classes. See the classification report for per-role metrics.")

    st.header("3. Classify resumes")
    mode = st.radio("Choose input method", ["Upload resume files", "Paste resume text"], horizontal=True)

    if mode == "Upload resume files":
        resume_files = st.file_uploader(
            "Upload resumes (.pdf, .docx, .txt)",
            type=["pdf", "docx", "txt"], accept_multiple_files=True, key="resume_uploads"
        )
        if resume_files and st.button("Classify uploaded resumes"):
            rows = []
            for file in resume_files:
                try:
                    text = extract_resume_text(file)
                    if not text.strip():
                        rows.append({"File name": file.name, "Predicted role": "No text extracted", "Confidence": None, "Status": "Check scanned PDF or file content"})
                        continue
                    model = st.session_state["resume_model"]
                    prediction = model.predict([text])[0]
                    probabilities = model.predict_proba([text])[0]
                    best_idx = probabilities.argmax()
                    top_indices = probabilities.argsort()[::-1][:3]
                    top_roles = " | ".join(
                        f"{model.classes_[i]} ({probabilities[i]:.1%})" for i in top_indices
                    )
                    rows.append({
                        "File name": file.name,
                        "Predicted role": prediction,
                        "Confidence": float(probabilities[best_idx]),
                        "Top 3 role matches": top_roles,
                        "Status": "OK"
                    })
                except Exception as exc:
                    rows.append({"File name": file.name, "Predicted role": "Error", "Confidence": None, "Top 3 role matches": "", "Status": str(exc)})
            results_df = pd.DataFrame(rows)
            st.dataframe(results_df, use_container_width=True)
            st.download_button(
                "Download predictions CSV", results_df.to_csv(index=False).encode("utf-8"),
                file_name="resume_classification_predictions.csv", mime="text/csv"
            )
    else:
        pasted_text = st.text_area("Paste the resume content", height=260, placeholder="Paste resume summary, skills, work experience, and education...")
        if st.button("Predict role from text"):
            if not pasted_text.strip():
                st.warning("Paste some resume text first.")
            else:
                model = st.session_state["resume_model"]
                probabilities = model.predict_proba([pasted_text])[0]
                order = probabilities.argsort()[::-1][:5]
                st.success(f"Predicted role: {model.predict([pasted_text])[0]}")
                top_df = pd.DataFrame({
                    "Role": [model.classes_[i] for i in order],
                    "Model score": [probabilities[i] for i in order]
                })
                top_df["Model score"] = top_df["Model score"].map(lambda x: f"{x:.1%}")
                st.dataframe(top_df, use_container_width=True)

    st.caption("Model scores are relative classifier outputs, not calibrated probabilities of job suitability. Predictions depend on the quality and balance of the labelled training data.")
else:
    st.info("Click **Train classifier** to see evaluation metrics and enable resume classification.")

st.divider()
st.caption("Portfolio project · TF-IDF + Logistic Regression · Use responsibly and review predictions before making hiring decisions.")
