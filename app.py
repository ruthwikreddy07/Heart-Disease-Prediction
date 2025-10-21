import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# ==============================
# 1. DATA PREPROCESSING
# ==============================
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Clean strings
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace(["", "?", "NA", "NaN", "nan"], np.nan, inplace=True)

    # Map boolean columns
    bool_map = {"TRUE": 1, "True": 1, True: 1, "FALSE": 0, "False": 0, False: 0, 1: 1, 0: 0}
    for bcol in ["fbs", "exang"]:
        if bcol in df.columns:
            df[bcol] = df[bcol].map(bool_map).fillna(0)

    # Map sex
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"Male": 1, "M": 1, "Female": 0, "F": 0}).fillna(0)

    # Categorical columns
    cat_cols = ["cp", "restecg", "slope", "thal"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip().fillna("unknown")
    if "thal" in df.columns:
        df["thal"] = df["thal"].replace({"reversable defect": "reversible defect"})

    # Numeric columns
    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

    # Target column
    if "num" in df.columns:
        df["num"] = (df["num"] > 0).astype(int)

    # Drop irrelevant columns
    df.drop(columns=['id', 'dataset'], inplace=True, errors='ignore')

    return df

df = load_and_preprocess_data("heart_disease_uci.csv")

# ==============================
# 2. MODEL TRAINING
# ==============================
y = df["num"]
X = df.drop("num", axis=1)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)

# ==============================
# 3. STREAMLIT APP
# ==============================
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("🩺 Heart Disease Prediction Application")
st.sidebar.header("Patient Data Input")

def get_user_input():
    cp_options = sorted(df['cp'].unique())
    thal_options = sorted(df['thal'].unique())
    slope_options = sorted(df['slope'].unique())
    restecg_options = sorted(df['restecg'].unique())

    # Age slider extended 0-100
    age = st.sidebar.slider("Age", 0, 100, int(df['age'].median()))
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type", cp_options)
    trestbps = st.sidebar.slider("Resting BP", int(df['trestbps'].min()), int(df['trestbps'].max()), int(df['trestbps'].median()))
    chol = st.sidebar.slider("Cholesterol", int(df['chol'].min()), int(df['chol'].max()), int(df['chol'].median()))
    fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0, 1], format_func=lambda x: "False" if x==0 else "True")
    restecg = st.sidebar.selectbox("Resting ECG", restecg_options)
    thalch = st.sidebar.slider("Max Heart Rate", 0, 250, int(df['thalch'].median()))  # max 250 for safety
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak = st.sidebar.slider("ST Depression", float(df['oldpeak'].min()), float(df['oldpeak'].max()), float(df['oldpeak'].median()))
    slope = st.sidebar.selectbox("Slope of ST Segment", slope_options)
    ca = st.sidebar.slider("Major Vessels Colored (ca)", int(df['ca'].min()), int(df['ca'].max()), int(df['ca'].median()))
    thal = st.sidebar.selectbox("Thalassemia", thal_options)

    return pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
        'fbs': [fbs], 'restecg': [restecg], 'thalch': [thalch], 'exang': [exang],
        'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })

user_input = get_user_input()
st.subheader("Patient Input Data")
st.write(user_input)

if st.button("Predict Heart Disease"):
    # Heart rate capping logic
    age_val = user_input['age'].iloc[0]
    thalch_val = user_input['thalch'].iloc[0]
    estimated_max_hr = 220 - age_val
    if thalch_val > estimated_max_hr + 10:
        capped_thalch = estimated_max_hr + 10
        user_input['thalch'] = capped_thalch
        st.warning(f"Heart rate adjusted to {capped_thalch} for physiological accuracy.", icon="⚠️")

    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("High Risk: Patient MAY have heart disease.", icon="💔")
    else:
        st.success("Low Risk: Patient likely does NOT have heart disease.", icon="❤️")

    st.subheader("Prediction Confidence")
    st.write(f"**{prediction_proba[0][prediction[0]]*100:.2f}%** confidence")
    st.write(f"Probability of No Disease (0): {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Probability of Disease (1): {prediction_proba[0][1]*100:.2f}%")
