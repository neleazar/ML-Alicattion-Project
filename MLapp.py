import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Title of the app
st.title("ML Application Project")

st.write("**By Eleazar Neamat**")
st.write("This project is a Python-based application built with Streamlit that allows users to perform machine learning classification tasks on datasets like Iris or Tips, featuring data preprocessing, model training, evaluation, and prediction with an interactive interface.")
# Dataset selection
st.header("Dataset Selection")
dataset_name = st.selectbox("Select Dataset", ["Iris", "Tips"])
if dataset_name == "Iris":
    df = sns.load_dataset("iris")
elif dataset_name == "Tips":
    df = sns.load_dataset("tips")

# Display dataset
st.write("### Loaded Dataset:")
st.dataframe(df)

# Preprocessing
st.header("Preprocessing")
# Handle missing values
if df.isnull().sum().sum() > 0:
    st.write("Handling missing values...")
    df = df.dropna()  # Drop missing values

# Encode categorical variables
categorical_columns = df.select_dtypes(include='object').columns
if len(categorical_columns) > 0:
    st.write("Encoding categorical variables...")
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
    st.write("### Dataset after Encoding:")
    st.dataframe(df)
else:
    st.write("No categorical data to encode.")

# Feature and target selection
st.header("Feature and Target Selection")
features = st.multiselect("Select Features", options=df.columns, default=df.columns[:-1])
target = st.selectbox("Select Target Variable", options=[col for col in df.columns if col not in features])

# Data splitting
st.header("Data Splitting")
test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

st.write(f"Training Set Size: {len(X_train)} rows")
st.write(f"Test Set Size: {len(X_test)} rows")

# Model selection
st.header("Model Selection")
model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest Classifier"])
if model_name == "Logistic Regression":
    model = LogisticRegression()
elif model_name == "Random Forest Classifier":
    model = RandomForestClassifier()

# Train model button
if st.button("Train Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save the trained model and label encoder in Streamlit session state
    st.session_state['trained_model'] = model
    st.session_state['label_encoder'] = encoder if target in categorical_columns else None

    # Display metrics
    st.write("### Model Performance:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Prediction
st.header("Make Predictions")
if 'trained_model' in st.session_state:
    input_data = {}
    for feature in features:
        value = st.number_input(f"Enter value for {feature}", value=float(np.mean(df[feature])))
        input_data[feature] = value

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        numeric_prediction = st.session_state['trained_model'].predict(input_df)[0]

        # Convert numeric prediction to class name if a label encoder is available
        if 'label_encoder' in st.session_state and st.session_state['label_encoder']:
            prediction = st.session_state['label_encoder'].inverse_transform([numeric_prediction])[0]
        else:
            prediction = numeric_prediction

        st.write(f"### Prediction Result: {prediction}")
else:
    st.write("Train the model first before making predictions.")
