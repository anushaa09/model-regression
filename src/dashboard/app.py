import os
import streamlit as st
import pandas as pd
import requests

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)


st.title(
    "🤖 Model Regression Detector"
)


st.write(
    "Compare baseline and candidate ML model versions "
    "and detect performance regressions."
)

if st.button("Run Evaluation"):

    response = requests.post(
        f"{API_URL}/run-eval"
    )

    if response.status_code == 200:

        data = response.json()

        result = data["result"]

        if result["status"] == "FAIL":

            st.error(
                "🚨 Regression Detected"
            )

        else:

            st.success(
                "✅ No Regression Detected"
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Baseline Accuracy",
                f"{result['baseline_metric']:.2%}"
            )

        with col2:

            st.metric(
                "Candidate Accuracy",
                f"{result['candidate_metric']:.2%}"
            )

        st.write(
            "p-value:",
            result["p_value"]
        )

        st.json(
            result["details"]
        )


st.subheader(
    "Evaluation History"
)


try:

    response = requests.get(
        f"{API_URL}/history"
    )

    if response.status_code == 200:

        history = response.json()

        if history:

            df = pd.DataFrame(
                history
            )

            st.dataframe(
                df,
                use_container_width=True
            )

except Exception:

    st.info(
        "Start the FastAPI server "
        "to view history."
    )