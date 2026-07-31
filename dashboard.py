import streamlit as st
import pandas as pd

st.set_page_config(page_title="RAG Calibration Dashboard", layout="wide")

RESULTS_FILES = {
    "llama3.2:3b": "results_llama3_2-3b.csv",
    "mistral:7b": "results_mistral-7b.csv",
    "llama-3.1-8b-instant": "results_llama-3_1-8b-instant.csv",
}
SCORES_FILE = "scores.csv"

@st.cache_data
def load_data():
    scores = pd.read_csv(SCORES_FILE)
    all_rows = []
    for model, fname in RESULTS_FILES.items():
        try:
            df = pd.read_csv(fname)
            df["confidence_clean"] = df["confidence"].astype(str).str.extract(r"^(HIGH|MEDIUM|LOW)")
            df["model"] = model
            all_rows.append(df)
        except FileNotFoundError:
            st.warning(f"Could not find {fname} — skipping {model}.")
    combined = pd.concat(all_rows, ignore_index=True)
    merged = combined.merge(scores, on=["model", "id"], how="left")
    return merged

df = load_data()

st.title("RAG Calibration & Reliability Dashboard")
st.caption("Testing whether self-reported LLM confidence predicts actual answer correctness.")

# --- Headline stats ---
overall_acc = df["correct"].mean() * 100
high_acc = df[df["confidence_clean"] == "HIGH"]["correct"].mean() * 100
low_acc = df[df["confidence_clean"] == "LOW"]["correct"].mean() * 100

col1, col2, col3 = st.columns(3)
col1.metric("Overall Accuracy", f"{overall_acc:.1f}%")
col2.metric("Accuracy when HIGH confidence", f"{high_acc:.1f}%")
col3.metric("Accuracy when LOW confidence", f"{low_acc:.1f}%", delta=f"{low_acc - high_acc:+.1f} pts vs HIGH")

st.markdown("---")

# --- Chart: HIGH vs LOW accuracy per model ---
st.subheader("Confidence Reliability by Model")
chart_data = (
    df[df["confidence_clean"].isin(["HIGH", "LOW"])]
    .groupby(["model", "confidence_clean"])["correct"]
    .mean()
    .mul(100)
    .reset_index()
    .pivot(index="model", columns="confidence_clean", values="correct")
)
st.bar_chart(chart_data)
st.caption("If confidence were well-calibrated, HIGH bars would be taller than LOW bars. Here, the opposite holds for every model.")

# --- Category breakdown ---
st.subheader("Accuracy by Question Category")
cat_data = df.groupby("category")["correct"].mean().mul(100).reset_index()
st.bar_chart(cat_data.set_index("category"))

# --- Failure gallery ---
st.subheader("Notable Failures: Confident but Wrong")
failures = df[(df["confidence_clean"] == "HIGH") & (df["correct"] == 0)]
for _, row in failures.iterrows():
    with st.expander(f"[{row['model']}] {row['question']}"):
        st.write(f"**Answer given:** {row['answer']}")
        st.write(f"**Evidence cited:** {row['evidence']}")
        st.write(f"**Retrieved source:** {row['retrieved_sources']}")
        st.write(f"**Confidence:** HIGH — but scored incorrect")

# --- Raw data explorer ---
st.subheader("Explore Raw Results")
selected_model = st.selectbox("Filter by model", ["All"] + list(RESULTS_FILES.keys()))
display_df = df if selected_model == "All" else df[df["model"] == selected_model]
st.dataframe(display_df[["model", "id", "category", "question", "answer", "confidence_clean", "correct"]])