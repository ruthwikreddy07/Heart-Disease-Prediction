import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# =====================================================================================
# 1. DATA LOADING AND PREPROCESSING
# =====================================================================================

@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads and cleans the heart disease dataset."""
    df = pd.read_csv(file_path)
    # Applying all the robust cleaning steps from our previous work
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace(["", "?", "NA", "NaN", "nan"], np.nan, inplace=True)
    bool_map = {"TRUE": 1, "True": 1, True: 1, "FALSE": 0, "False": 0, False: 0, 1: 1, 0: 0}
    for col in ["fbs", "exang"]:
        if col in df.columns: df[col] = df[col].map(bool_map)
    if "sex" in df.columns: df["sex"] = df["sex"].map({"Male": 1, "M": 1, "Female": 0, "F": 0})
    cat_cols = ["cp", "restecg", "slope", "thal", "dataset"]
    for col in cat_cols:
        if col in df.columns: df[col] = df[col].astype("string").str.lower().str.strip()
    if "thal" in df.columns: df["thal"] = df["thal"].replace({"reversable defect": "reversible defect"})
    num_cols = ["id", "age", "trestbps", "chol", "thalch", "oldpeak", "ca", "num"]
    for col in num_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["trestbps", "chol", "thalch", "oldpeak"]:
        if col in df.columns: df[col].fillna(df[col].median(), inplace=True)
    if "ca" in df.columns: df["ca"].fillna(df["ca"].mode().iloc[0], inplace=True)
    for col in ["fbs", "exang"]:
        if col in df.columns: df[col].fillna(df[col].mode().iloc[0], inplace=True)
    for col in cat_cols:
        if col in df.columns: df[col].fillna(df[col].mode().iloc[0], inplace=True)
    if "num" in df.columns: df["num"] = (df["num"] > 0).astype(int)
    df.drop(columns=['id', 'dataset'], inplace=True, errors='ignore')
    return df

# Load data
df = load_and_preprocess_data("heart_disease_uci.csv")
y = df["num"]
X = df.drop("num", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================================
# 2. UI AND DYNAMIC MODEL TRAINING
# =====================================================================================
st.set_page_config(page_title="Comprehensive ML Predictor", layout="wide")
st.title("❤️‍🩹 Heart Disease Prediction with Cluster-Enhanced AI")
st.markdown("""
This application demonstrates a powerful **"Cluster-then-Classify"** strategy.
1.  First, an **unsupervised clustering algorithm** finds natural subgroups in the patient data.
2.  The discovered patient group is then used as a **new feature** to help a **supervised classification algorithm** make a more accurate prediction.
""")

st.sidebar.title("⚙️ Model Configuration")
st.sidebar.header("Step 1: Choose a Clustering Algorithm")
cluster_choice = st.sidebar.selectbox("Algorithm to discover patient subgroups",
                                      ["K-Means", "Agglomerative Clustering", "DBSCAN"])

st.sidebar.header("Step 2: Choose a Classification Algorithm")
classifier_choice = st.sidebar.selectbox("Algorithm for final prediction",
                                         ["Random Forest", "Gradient Boosting", "AdaBoost"])

st.sidebar.header("Step 3: Input Patient Data")
def user_input_features():
    # User input widget code remains the same...
    age = st.sidebar.slider("Age", 20, 80, 50)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=sorted(df['cp'].unique()))
    trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (chol)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
    restecg = st.sidebar.selectbox("Resting ECG (restecg)", options=sorted(df['restecg'].unique()))
    thalch = st.sidebar.slider("Max Heart Rate (thalach)", 70, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope", options=sorted(df['slope'].unique()))
    ca = st.sidebar.slider("Major vessels colored (ca)", 0, 4, 0)
    thal = st.sidebar.selectbox("Thalassemia (thal)", options=sorted(df['thal'].unique()))
    user_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs],
        'restecg': [restecg], 'thalch': [thalch], 'exang': [exang], 'oldpeak': [oldpeak],
        'slope': [slope], 'ca': [ca], 'thal': [thal]
    })
    return user_data
input_df = user_input_features()

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['string', 'object']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

if st.button(f"Run Prediction using {cluster_choice} + {classifier_choice}"):
    with st.spinner("Training models and making prediction..."):
        # --- MODEL DEFINITION ---
        if cluster_choice == "K-Means":
            clusterer = KMeans(n_clusters=4, random_state=42, n_init=10)
        elif cluster_choice == "Agglomerative Clustering":
            clusterer = AgglomerativeClustering(n_clusters=4)
        else: # DBSCAN
            clusterer = DBSCAN(eps=2.5, min_samples=5)

        if classifier_choice == "Random Forest":
            classifier = RandomForestClassifier(random_state=42)
        elif classifier_choice == "Gradient Boosting":
            classifier = GradientBoostingClassifier(random_state=42)
        else: # AdaBoost
            classifier = AdaBoostClassifier(random_state=42)

        # --- MODEL TRAINING ---
        # Build and train a simple pipeline with just the classifier for the fallback case
        base_classifier_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
        base_classifier_pipeline.fit(X_train, y_train)

        # --- PREDICTION LOGIC WITH BUG FIX ---
        cluster_name = "N/A (Prediction mode not supported by this algorithm)"
        
        # The 'cluster-then-classify' only works if the clusterer has a .predict method
        if hasattr(clusterer, 'predict'):
            # Build and train the full cluster-enhanced pipeline
            # 1. Preprocess data
            X_train_processed = preprocessor.fit_transform(X_train)
            clusterer.fit(X_train_processed)
            # 2. Add cluster labels as a new feature
            train_cluster_labels = clusterer.labels_.reshape(-1, 1)
            X_train_enhanced = np.hstack((X_train_processed, train_cluster_labels))
            # 3. Train classifier on enhanced data
            classifier.fit(X_train_enhanced, y_train)
            
            # 4. Process user input
            input_processed = preprocessor.transform(input_df)
            input_cluster_label = clusterer.predict(input_processed).reshape(-1, 1)
            input_enhanced = np.hstack((input_processed, input_cluster_label))
            
            # 5. Make final prediction and get cluster name
            prediction = classifier.predict(input_enhanced)
            prediction_proba = classifier.predict_proba(input_enhanced)
            cluster_name = f"Group {input_cluster_label[0][0]}"
        else:
            # Fallback for Agglomerative and DBSCAN
            prediction = base_classifier_pipeline.predict(input_df)
            prediction_proba = base_classifier_pipeline.predict_proba(input_df)
        
        # --- DISPLAY RESULTS ---
        st.header("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Discovered Patient Subgroup")
            st.metric("Patient belongs to:", cluster_name)
        
        with col2:
            st.info("Final Prediction")
            if prediction[0] == 1:
                st.error("High Risk: Heart Disease Likely", icon="💔")
            else:
                st.success("Low Risk: Heart Disease Unlikely", icon="❤️")
        
        st.subheader("Prediction Confidence")
        confidence = prediction_proba[0].max() * 100
        st.progress(int(confidence))
        st.metric(label="Confidence", value=f"{confidence:.2f}%")