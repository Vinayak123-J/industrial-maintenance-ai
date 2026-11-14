from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. PATHS AND PAGE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_PATH = BASE_DIR / "models" / "failure_model.pkl"
IMPORTANCE_PATH = BASE_DIR / "reports" / "feature_importance.csv"

st.set_page_config(
    page_title="Industrial Maintenance AI",
    page_icon="🏭",
    layout="wide",
)

# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        saved = pickle.load(file)

    return saved["model"], saved["features"]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model, features = load_model()
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the model or dataset: {exc}")
    st.stop()


# --------------------------------------------------
# 3. REPRODUCE THE TEST SPLIT FOR EVALUATION
# --------------------------------------------------

@st.cache_data
def prepare_test_data():
    data = pd.read_csv(DATA_PATH)

    target = "Machine failure"

    drop_columns = [
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    data = data.drop(columns=drop_columns, errors="ignore")

    X = data.drop(columns=[target])
    y = data[target]

    X = pd.get_dummies(
        X,
        columns=["Type"],
        dtype=int,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_test, y_test


# --------------------------------------------------
# 4. DASHBOARD HEADER
# --------------------------------------------------

st.title("🏭 AI-Based Industrial Maintenance")
st.subheader("Predictive Maintenance & Process Optimization")

st.write(
    "Predict machine failure risk, inspect model performance, "
    "understand important features, and review maintenance guidance."
)

st.info(
    "Educational demonstration using the AI4I 2020 simulated dataset. "
    "Model predictions and recommendations are not certified industrial "
    "safety decisions."
)

# Four sections in the dashboard
tab_predict, tab_performance, tab_features, tab_maintenance = st.tabs(
    [
        "🔮 Failure Prediction",
        "📊 Model Performance",
        "📈 Feature Importance",
        "🛠️ Maintenance Guidance",
    ]
)


# --------------------------------------------------
# 5. FAILURE PREDICTION
# --------------------------------------------------

with tab_predict:
    st.header("Machine Operating Parameters")

    with st.form("machine_form"):
        col1, col2 = st.columns(2)

        with col1:
            machine_type = st.selectbox(
                "Machine Type",
                ["L", "M", "H"],
                index=1,
            )

            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0,
                step=0.1,
            )

            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=310.0,
                step=0.1,
            )

            rotational_speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=0,
                max_value=10000,
                value=1500,
            )

        with col2:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=500.0,
                value=40.0,
                step=0.1,
            )

            tool_wear = st.number_input(
                "Tool Wear (minutes)",
                min_value=0,
                max_value=1000,
                value=100,
            )

        submitted = st.form_submit_button(
            "Predict Machine Failure",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=features,
        )

        input_data = input_data.astype({
    "Air temperature [K]": "float64",
    "Process temperature [K]": "float64",
    "Rotational speed [rpm]": "float64",
    "Torque [Nm]": "float64",
    "Tool wear [min]": "float64"
})

        measurements = {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }

        for column, value in measurements.items():
            if column in input_data.columns:
                input_data.loc[0, column] = value

        type_column = f"Type_{machine_type}"

        if type_column in input_data.columns:
            input_data.loc[0, type_column] = 1

        try:
            prediction = int(model.predict(input_data)[0])
            probability = float(
                model.predict_proba(input_data)[0][1]
            )

            st.session_state["last_prediction"] = {
                "prediction": prediction,
                "probability": probability,
                "machine_type": machine_type,
                "air_temp": air_temp,
                "process_temp": process_temp,
                "rotational_speed": rotational_speed,
                "torque": torque,
                "tool_wear": tool_wear,
            }

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("last_prediction")

    if result:
        st.divider()
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        col1.metric(
            "Estimated Failure Probability",
            f"{result['probability']:.2%}",
        )

        col2.metric(
            "Model Classification",
            "Failure predicted"
            if result["prediction"] == 1
            else "No failure predicted",
        )

        st.progress(result["probability"])

        if result["prediction"] == 1:
            st.error(
                "The model predicts a possible failure. "
                "Arrange an appropriate inspection and follow "
                "your established maintenance procedures."
            )
        else:
            st.success(
                "The model does not predict a failure for these inputs. "
                "Continue routine monitoring and scheduled maintenance."
            )

        st.caption(
            "A low predicted probability does not guarantee safety. "
            "The model should be validated on representative real-world "
            "machine data before operational use."
        )


# --------------------------------------------------
# 6. MODEL PERFORMANCE
# --------------------------------------------------

with tab_performance:
    st.header("Model Evaluation")

    try:
        X_test, y_test = prepare_test_data()

        # Ensure the test columns match the model's saved feature list.
        X_test = X_test.reindex(
            columns=features,
            fill_value=0,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )
        auc = roc_auc_score(y_test, y_prob)

        st.caption(
            "Metrics below use the reproducible 20% stratified test split. "
            "They are meaningful only if the loaded model was trained "
            "using the same split and preprocessing."
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Accuracy", f"{accuracy:.2%}")
        c2.metric("Precision", f"{precision:.2%}")
        c3.metric("Recall", f"{recall:.2%}")
        c4.metric("F1-score", f"{f1:.2%}")
        c5.metric("ROC-AUC", f"{auc:.3f}")

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Failure", "Failure"],
        )

        display.plot(
            ax=ax,
            values_format="d",
            colorbar=False,
        )

        ax.set_title("Machine Failure Classification")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.write(
            "**Interpretation:** Recall measures how many actual "
            "failures the model detects. Precision measures how often "
            "a predicted failure is actually a failure in the test data."
        )

        st.caption(
            "Because failures are rare, accuracy alone can be misleading. "
            "Review failure-class precision and recall carefully."
        )

    except Exception as exc:
        st.error(f"Could not evaluate the model: {exc}")


# --------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------

with tab_features:
    st.header("Important Machine Features")

    try:
        importance = pd.read_csv(IMPORTANCE_PATH)

        required_columns = {"Feature", "Importance"}

        if not required_columns.issubset(importance.columns):
            st.error(
                "The feature importance CSV must contain "
                "'Feature' and 'Importance' columns."
            )
        else:
            importance = importance.sort_values(
                "Importance",
                ascending=False,
            )

            top_n = st.slider(
                "Number of features to display",
                min_value=3,
                max_value=min(15, len(importance)),
                value=min(10, len(importance)),
            )

            chart_data = (
                importance.head(top_n)
                .sort_values("Importance")
                .set_index("Feature")
            )

            st.bar_chart(chart_data["Importance"])

            st.subheader("Feature Importance Table")
            st.dataframe(
                importance.head(top_n),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Random Forest feature importance indicates how the model "
                "uses features; it does not prove causation or guarantee "
                "that changing a feature will prevent failure."
            )

    except FileNotFoundError:
        st.warning(
            "Feature importance report not found. Run train_model.py first."
        )
    except Exception as exc:
        st.error(f"Could not display feature importance: {exc}")


# --------------------------------------------------
# 8. MAINTENANCE AND PROCESS GUIDANCE
# --------------------------------------------------

with tab_maintenance:
    st.header("Maintenance & Process Guidance")

    st.write(
        "The following checks compare entered measurements with the "
        "distribution of the AI4I 2020 dataset. They are screening "
        "suggestions, not validated engineering limits."
    )

    try:
        data = df

        if "last_prediction" in st.session_state:
            r = st.session_state["last_prediction"]

            checks = [
                (
                    "Air temperature [K]",
                    r["air_temp"],
                    "Review air-temperature sensor readings and cooling conditions.",
                ),
                (
                    "Process temperature [K]",
                    r["process_temp"],
                    "Review process-temperature trends and thermal control.",
                ),
                (
                    "Rotational speed [rpm]",
                    r["rotational_speed"],
                    "Inspect speed-control settings, vibration, and operating logs.",
                ),
                (
                    "Torque [Nm]",
                    r["torque"],
                    "Review load conditions and inspect mechanical components if indicated.",
                ),
                (
                    "Tool wear [min]",
                    r["tool_wear"],
                    "Check the tool inspection schedule and wear records.",
                ),
            ]

            recommendations = []

            for column, value, message in checks:
                if column not in data.columns:
                    continue

                # High-end reference point from the dataset, not a safety limit.
                reference = float(data[column].quantile(0.95))

                if value > reference:
                    recommendations.append(
                        (column, value, reference, message)
                    )

            if r["prediction"] == 1:
                st.warning(
                    "The model predicted a possible failure. Prioritize "
                    "review by qualified maintenance personnel."
                )

            if recommendations:
                st.subheader("Measurements to Review")

                for column, value, reference, message in recommendations:
                    st.warning(
                        f"**{column}: {value:g}** is above the dataset's "
                        f"95th percentile reference ({reference:.2f}). "
                        f"{message}"
                    )
            else:
                st.success(
                    "No entered measurement exceeds the selected "
                    "95th-percentile screening references. This does not "
                    "mean the machine is safe or free from wear."
                )

            st.subheader("Suggested Maintenance Workflow")

            st.markdown(
                """
                1. Review sensor readings and machine operating logs.
                2. Check maintenance history and tool-wear records.
                3. Inspect relevant components according to the manufacturer's
                   procedures and your organization's safety requirements.
                4. Record inspection findings and compare them with future
                   measurements.
                5. Use verified engineering limits and real machine data
                   before making operational decisions.
                """
            )

        else:
            st.info(
                "Make a prediction in the Failure Prediction tab first "
                "to receive measurement-specific guidance."
            )

    except Exception as exc:
        st.error(f"Could not generate maintenance guidance: {exc}")


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Industrial Maintenance and Process Optimization System | "
    "Academic demonstration"
)