"""
dashboard/app.py

Streamlit dashboard that reads all experiment runs directly from MLflow
tracking and plots:
  - Centralized vs Federated-IID vs Federated-Non-IID vs Federated-with-Dropout
    accuracy curves
  - (if present) privacy-utility trade-off curve from DP sweeps

Run:
    streamlit run dashboard/app.py
"""

import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Federated Learning — Results Dashboard", layout="wide")

EXPERIMENT_NAME = "federated-learning-sim"


@st.cache_data(ttl=30)
def load_runs():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame(), {}

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    metric_histories = {}
    run_labels = {}
    for run in runs:
        run_id = run.info.run_id
        label = run.data.tags.get("mlflow.runName", run_id[:8])
        run_labels[run_id] = label
        history = client.get_metric_history(run_id, "global_accuracy")
        if history:
            metric_histories[label] = pd.Series(
                {m.step: m.value for m in history}
            ).sort_index()

    return pd.DataFrame(metric_histories), run_labels


def main():
    st.title("Federated Learning Simulation — Results Dashboard")
    st.caption("FedAvg vs centralized training, tracked via MLflow")

    df, _ = load_runs()

    if df.empty:
        st.warning(
            "No MLflow runs found yet. Run `baseline/centralized_train.py` and "
            "`fl/server.py` (with different `--partition` / `--fraction-fit` args) first."
        )
        return

    st.subheader("Accuracy over rounds / epochs")
    st.line_chart(df)

    st.subheader("Final accuracy by run")
    final_acc = df.ffill().iloc[-1].sort_values(ascending=False)
    st.bar_chart(final_acc)

    with st.sidebar:
        st.header("Summary of Findings")
        st.markdown(
            """
            **What this compares:**
            - *Centralized baseline* — upper-bound accuracy with all data pooled
            - *Federated IID* — FedAvg with evenly distributed client data
            - *Federated Non-IID* — FedAvg with Dirichlet-skewed client data (harder)
            - *Federated + dropout* — only a fraction of clients participate each round

            **Expected pattern:** centralized > federated-IID > federated-non-IID,
            with dropout runs converging more slowly than full-participation runs.
            Edit this panel with your own numbers once you've run the experiments.
            """
        )


if __name__ == "__main__":
    main()
