
# Step 9.3.1: Import required libraries
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Step 9.3.2: Configure the website
st.set_page_config(
    page_title="Industrial Fault Detection",
    page_icon="⚙️",
    layout="wide"
)

# Step 9.3.3: Define project paths
BASE_DIR = Path(__file__).resolve().parent

# Step 9.3.4: Load the dataset and trained model
@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "ai4i2020.csv")

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "models" / "random_forest_model.pkl")
    columns = joblib.load(BASE_DIR / "models" / "feature_columns.pkl")
    return model, columns

df = load_data()
model, feature_columns = load_model()

# Step 9.3.5: Display website header
st.title("⚙️ Industrial Equipment Fault Detection")

st.write(
    "A Machine Learning-based platform for analyzing "
    "industrial equipment conditions and detecting potential failures."
)

st.divider()

# Step 9.3.6: Display dataset summary
st.subheader("Equipment Monitoring Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Recorded Failures", int(df["Machine failure"].sum()))

with col3:
    failure_rate = df["Machine failure"].mean() * 100
    st.metric("Failure Rate", f"{failure_rate:.2f}%")

# Step 9.3.7: Display sample data
st.subheader("Equipment Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Step 9.3.8: Display current project status
st.info(
    "Baseline Random Forest model loaded successfully. "
    "Interactive fault prediction and additional analytics "
    "will be added next."
)


# Step 10.1.1: Create the fault prediction section
st.divider()
st.subheader("Machine Fault Prediction")

st.write(
    "Enter the operating conditions of an industrial machine "
    "to predict whether it is likely to experience a failure."
)

# Step 10.1.2: Create input fields
col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=310.0
    )

    rotational_speed = st.number_input(
        "Rotational Speed (RPM)",
        min_value=0,
        max_value=5000,
        value=1500
    )

with col2:
    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=300,
        value=50
    )

    product_type = st.selectbox(
        "Product Type",
        options=["L", "M", "H"]
    )

# Step 10.1.3: Create the prediction button
if st.button("Predict Machine Failure"):

    # Step 10.1.4: Prepare input data
    input_data = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type": product_type
    }])

    # Step 10.1.5: Encode product type
    input_encoded = pd.get_dummies(
        input_data,
        columns=["Type"],
        dtype=int
    )

    # Step 10.1.6: Match training feature columns
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Step 10.1.7: Make prediction
    prediction = model.predict(input_encoded)[0]

    failure_probability = model.predict_proba(
        input_encoded
    )[0][1]

    # Step 10.1.8: Display prediction results
    st.subheader("Prediction Results")

    if prediction == 1:
        st.error("Potential Machine Failure Detected!")
    else:
        st.success("Machine Operating Normally")

    st.metric(
        "Estimated Failure Score",
        f"{failure_probability * 100:.2f}%"
    )

    st.caption(
        "This is a model-generated estimate based on a "
        "synthetic dataset, not a validated real-world "
        "equipment failure probability."
    )
