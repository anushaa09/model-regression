import os

import pandas as pd
import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)

st.set_page_config(
    page_title="Model Regression Detector",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 Model Regression Detector")

st.caption(
    "ML model regression testing and performance monitoring"
)

st.markdown("---")


# --------------------------------------------------
# Run Evaluation
# --------------------------------------------------

if st.button("🚀 Run Model Evaluation", use_container_width=True):

    with st.spinner("Running model evaluation..."):

        try:

            response = requests.post(
                f"{API_URL}/run-eval",
                timeout=180
            )

            if response.status_code != 200:

                st.error(
                    f"Evaluation failed: HTTP {response.status_code}"
                )

            else:

                data = response.json()
                result = data["result"]

                st.session_state["result"] = result
                st.session_state["run_id"] = data["run_id"]

        except requests.exceptions.RequestException as e:

            st.error(
                f"Could not connect to the FastAPI backend.\n\n{e}"
            )


# --------------------------------------------------
# Display Evaluation Result
# --------------------------------------------------

if "result" in st.session_state:

    result = st.session_state["result"]

    details = result["details"]

    run_id = st.session_state["run_id"]


    # --------------------------------------------------
    # Status
    # --------------------------------------------------

    if result["status"] == "FAIL":

        st.error(
            f"🚨 REGRESSION DETECTED — Run #{run_id}"
        )

    else:

        st.success(
            f"✅ NO REGRESSION DETECTED — Run #{run_id}"
        )


    # --------------------------------------------------
    # Main Metrics
    # --------------------------------------------------

    st.subheader("Overall Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Baseline Accuracy",
            f"{details['baseline_accuracy']:.2%}"
        )

    with col2:

        st.metric(
            "Candidate Accuracy",
            f"{details['candidate_accuracy']:.2%}"
        )

    with col3:

        st.metric(
            "Accuracy Change",
            f"{-details['accuracy_drop']:.2%}"
        )


    # --------------------------------------------------
    # F1
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Baseline F1",
            f"{details['baseline_f1']:.2%}"
        )

    with col2:

        st.metric(
            "Candidate F1",
            f"{details['candidate_f1']:.2%}"
        )

    with col3:

        st.metric(
            "F1 Change",
            f"{-details['f1_drop']:.2%}"
        )


    # --------------------------------------------------
    # Statistical Test
    # --------------------------------------------------

    st.subheader("Statistical Significance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "McNemar p-value",
            f"{result['p_value']:.2e}"
        )

    with col2:

        if result["statistically_significant"]:

            st.warning(
                "Statistically significant difference"
            )

        else:

            st.success(
                "No statistically significant difference"
            )


    # --------------------------------------------------
    # Slice Analysis
    # --------------------------------------------------

    st.subheader("Age Slice Analysis")

    slices = details.get("slices", {})

    if slices:

        rows = []

        for group, metrics in slices.items():

            rows.append({
                "Age Group": group,
                "Samples": metrics["samples"],
                "Baseline Accuracy":
                    metrics["baseline_accuracy"],
                "Candidate Accuracy":
                    metrics["candidate_accuracy"],
                "Accuracy Drop":
                    metrics["accuracy_drop"],
                "Baseline F1":
                    metrics["baseline_f1"],
                "Candidate F1":
                    metrics["candidate_f1"],
                "F1 Drop":
                    metrics["f1_drop"]
            })

        slice_df = pd.DataFrame(rows)

        st.dataframe(
            slice_df.style.format({
                "Baseline Accuracy": "{:.2%}",
                "Candidate Accuracy": "{:.2%}",
                "Accuracy Drop": "{:.2%}",
                "Baseline F1": "{:.2%}",
                "Candidate F1": "{:.2%}",
                "F1 Drop": "{:.2%}"
            }),
            use_container_width=True
        )


    # --------------------------------------------------
    # Full JSON
    # --------------------------------------------------

    with st.expander("🔍 View Full Evaluation Details"):

        st.json(result)


# --------------------------------------------------
# Evaluation History
# --------------------------------------------------

st.markdown("---")

st.subheader("📊 Evaluation History")

try:

    response = requests.get(
        f"{API_URL}/history",
        timeout=30
    )

    if response.status_code == 200:

        history = response.json()

        if history:

            history_df = pd.DataFrame(history)

            st.dataframe(
                history_df,
                use_container_width=True
            )

        else:

            st.info("No evaluation runs yet.")

    else:

        st.warning(
            "Could not retrieve evaluation history."
        )

except requests.exceptions.RequestException:

    st.info(
        "FastAPI backend is currently unavailable."
    )