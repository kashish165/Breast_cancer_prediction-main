import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# --- STREAMLIT UI CONFIGURATION (MUST BE FIRST) ---
# FIX: Moved st.set_page_config to the very beginning as required by Streamlit.
st.set_page_config(
    page_title="Breast Cancer Risk Predictor",
    page_icon="🧬",
    layout="wide"
)

# --- CONFIGURATION AND SETUP ---

# 1. Load Model
# Assumes 'model.pkl' is in the same directory as this script.
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please ensure your pickled model file is in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. Re-fit Scaler (Crucial Step for Deployment)
# We must use the *exact* scaler fitted during training. Since we don't have the saved scaler.pkl,
# we re-fit a new scaler using the original data to ensure correct transformation.
@st.cache_resource
def get_scaler():
    """Loads the dataset and fits the StandardScaler."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler

scaler = get_scaler()
data_features = load_breast_cancer()
feature_names = data_features.feature_names

# Initialize session state for all features
# This ensures that all feature keys are present before any widget tries to read them.
for feature in feature_names:
    if feature not in st.session_state:
        st.session_state[feature] = 0.0


# --- STREAMLIT UI & LOGIC ---

st.title("🧬 Breast Cell Analysis: Risk Prediction Tool")
st.markdown(
    """
    This tool uses a Decision Tree model to predict the likelihood of a cell mass being
    **Malignant (Cancerous)** or **Benign (Non-Cancerous)** based on 30 geometric and
    textural properties measured from digitized images.

    **Disclaimer:** This is a demonstration model for educational purposes only and should
    NEVER be used for actual medical diagnosis. Consult a qualified physician for any health concerns.
    """
)
st.divider()

# Organize inputs into three main categories (Mean, Standard Error, Worst/Largest)
col1, col2, col3 = st.columns(3)

# Helper function to create sliders
def create_sliders(column, features_list, header):
    with column:
        st.subheader(header)
        for feature in features_list:
            # Determine min/max based on the feature name, using sensible defaults
            # (In a real app, min/max should come from the training data statistics)
            min_val = 0.0
            # np.where returns a tuple, [0][0] extracts the index
            feature_index = np.where(feature_names == feature)[0][0] 
            max_val = data_features.data[:, feature_index].max() * 1.1 
            
            # If max_val is extremely small (like for some 'se' features), use a reasonable upper bound
            if max_val < 0.5:
                max_val = 1.0 
            
            step = 0.01

            # Make labels user-friendly
            display_name = feature.replace('_', ' ').replace('mean', '(Average)').replace('se', '(Std Error)').replace('worst', '(Largest/Worst)').title()

            # The slider now stores its value directly into st.session_state using the feature name as the key.
            # We initialize the slider value to a default (0.0 or the stored session value)
            st.slider(
                display_name,
                min_value=min_val,
                max_value=max_val,
                value=st.session_state[feature], # Use session state for current value
                step=step,
                key=feature, # The key stores the value in st.session_state[feature]
                help=f"The input value for the '{feature}' feature."
            )
            

# Split features into 3 groups
mean_features = [f for f in feature_names if 'mean' in f]
se_features = [f for f in feature_names if 'se' in f]
worst_features = [f for f in feature_names if 'worst' in f]

# Populate the columns with sliders
create_sliders(col1, mean_features, "1. Average Measurements ('Mean')")
create_sliders(col2, se_features, "2. Standard Error ('SE')")
create_sliders(col3, worst_features, "3. Largest Measurements ('Worst')")


# --- PREDICTION BUTTON & RESULT ---

st.divider()

if st.button("Analyze Cell Characteristics", type="primary", use_container_width=True):
    # 1. Prepare input array by pulling data directly from session state (30 features)
    # Ensure the order of features matches the trained model
    input_list = [st.session_state[feature] for feature in feature_names]
    input_array = np.array(input_list).reshape(1, -1)

    # 2. Scale the input (using the re-fitted scaler on 30 features)
    scaled_input = scaler.transform(input_array) # Shape: (1, 30)
    
    # 3. CRITICAL FIX: Removed the logic that was inserting the 31st dummy feature.
    # The input is now consistently 30 features, matching the new model.pkl.

    # 4. Predict (model expects 30 features, scaled_input is (1, 30))
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    st.subheader("🔬 Prediction Result")

    # 5. Display result clearly for a layperson
    if prediction == 1:
        st.error(
            f"**Malignant Risk Detected**"
            f"\n\nBased on the entered data, the model predicts a **Malignant (Cancerous)** result."
            f" (Confidence: **{prediction_proba[1]*100:.2f}%**)"
        )
        st.warning("Immediate medical follow-up is highly recommended.")
    else:
        st.success(
            f"**Benign Risk Detected**"
            f"\n\nBased on the entered data, the model predicts a **Benign (Non-Cancerous)** result."
            f" (Confidence: **{prediction_proba[0]*100:.2f}%**)"
        )
        st.info("The characteristics appear non-cancerous according to the model.")

    st.markdown("---")
    st.caption("Review the input values and prediction confidence above.")