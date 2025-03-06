import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Streamlit App Configuration
st.set_page_config(page_title="FeatureForge - Predictive Modeling", layout="wide")

# Sidebar for Section Navigation
section = st.sidebar.radio("Select Section", ["Home", "EDA", "Preprocessing", "Split Data",
                                               "Train Model", "Evaluate Model", "Prediction",
                                               "Save/Load Model", "Feature Importance", "Download Results"])

# 1. Home Page and File Upload
if section == "Home":
    st.title("Welcome to FeatureForge 🚀")
    st.subheader("Automated, end-to-end predictive modeling")

    st.markdown("""
    ### Upload. Train. Predict.
    - 🛠️ **No Code, Just Results**: Streamline your machine learning workflow without writing a single line of code.
    - 📊 **Exploratory Data Analysis**: Gain insights into your data with visualizations and key statistics.
    - 🤖 **AutoML**: Leverage state-of-the-art algorithms to train models and get predictions instantly.
    - 🔍 **Feature Importance**: Understand which features drive your model’s decisions.
    
    🔼 **Get started by uploading your dataset!**
    """)
    
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df  # Store the uploaded dataframe in session state
        st.write("Preview of the uploaded dataset:")
        st.dataframe(df)

# 2. EDA (Exploratory Data Analysis) Section
if section == "EDA":
    if 'df' in st.session_state:
        df = st.session_state.df
        st.title("Exploratory Data Analysis")
        
        # Display Basic Information
        st.subheader("Dataset Information")
        st.write(df.info())
        
        # Display Basic Descriptive Statistics
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        
        # Correlation Heatmap
        if st.checkbox("Show Correlation Heatmap"):
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(11,5))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        # Pair Plot (for small datasets)
        if st.checkbox("Show Pair Plot (for numerical columns only)"):
            st.subheader("Pair Plot")
            sns.pairplot(df.select_dtypes(include=['int64', 'float64']),height=2.5)
            st.pyplot()

        # Distribution Plots
        st.subheader("Distribution of Each Feature")
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            st.write(f"Distribution for {column}")
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Please upload a dataset first.")

# 3. Data Preprocessing
if section == "Preprocessing":
    if 'df' in st.session_state:
        df = st.session_state.df
        target_variable = st.selectbox("Select the Target Variable", options=df.columns)

        if st.button("Preprocess Data"):
            # Separate features and target
            X = df.drop(columns=[target_variable])
            y = df[target_variable]
            st.session_state.X = X
            st.session_state.y = y
            
            # Identify numerical and categorical features
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns

            # Create preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)])

            # Apply preprocessing
            X_preprocessed = preprocessor.fit_transform(X)
            st.session_state.X_preprocessed = X_preprocessed

            st.success("Data preprocessing completed!")
            st.write("Preprocessed Data:")
            st.dataframe(X_preprocessed)
    else:
        st.warning("Please upload a dataset first.")

# 4. Data Splitting
if section == "Split Data":
    if 'X' in st.session_state and 'y' in st.session_state:
        X, y = st.session_state.X, st.session_state.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        st.write("Train and Test Data Split:")
        st.write(f"X_train shape: {X_train.shape}")
        st.write(f"X_test shape: {X_test.shape}")
        st.write(f"y_train shape: {y_train.shape}")
        st.write(f"y_test shape: {y_test.shape}")

        # Visualize Data Distribution
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(y_train, ax=ax[0], bins=30, kde=True)
        ax[0].set_title("Distribution of Training Target Variable")
        sns.histplot(y_test, ax=ax[1], bins=30, kde=True)
        ax[1].set_title("Distribution of Test Target Variable")
        st.pyplot(fig)

        st.success("Data split completed!")
    else:
        st.warning("Please preprocess the data first.")

# 5. Model Training
if section == "Train Model":
    if 'X_train' in st.session_state and 'y_train' in st.session_state:
        model_type = st.radio("Choose Model Type", options=['RandomForest', 'LogisticRegression'])

        if st.button("Train Model"):
            X_train, y_train = st.session_state.X_train, st.session_state.y_train
            if model_type == 'RandomForest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'LogisticRegression':
                model = LogisticRegression()

            model.fit(X_train, y_train)
            st.session_state.model = model  # Store the trained model in session state
            
            st.success(f"{model_type} trained successfully!")

            # Display model details
            st.write(f"Model: {model_type}")
            st.write("Training complete.")
    else:
        st.warning("Please split the data first.")

# 6. Model Evaluation
if section == "Evaluate Model":
    if 'model' in st.session_state:
        model = st.session_state.model
        X_test, y_test = st.session_state.X_test, st.session_state.y_test

        if st.button("Evaluate Model"):
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc:.4f}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Visualize Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(12,12))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
    else:
        st.warning("Please train the model first.")

# 7. Prediction on New Data
if section == "Prediction":
    st.header("Predict the Target Variable")
    
    if "model" in st.session_state and "X" in st.session_state:
        # Form for user input
        with st.form(key='prediction_form'):
            feature_inputs = {}
            for feature in st.session_state.X.columns:
                feature_inputs[feature] = st.text_input(f"Enter value for {feature}")

            submitted = st.form_submit_button("Make Prediction")
            if submitted:
                try:
                    input_data = {feature: float(value) for feature, value in feature_inputs.items()}
                    input_df = pd.DataFrame([input_data])

                    prediction = st.session_state.model.predict(input_df)
                    st.write(f"Predicted class: {prediction[0]}")
                except ValueError:
                    st.error("Please ensure all features are valid numbers.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please train the model before making predictions.")

# 8. Save/Load Model
if section == "Save/Load Model":
    if 'model' in st.session_state:
        if st.button("Save Model"):
            joblib.dump(st.session_state.model, "trained_model.pkl")
            st.success("Model saved successfully!")

        if st.button("Load Model"):
            model = joblib.load("trained_model.pkl")
            st.session_state.model = model
            st.success("Model loaded successfully!")
    else:
        st.warning("Please train the model first.")

# 9. Feature Importance
if section == "Feature Importance":
    if 'model' in st.session_state and isinstance(st.session_state.model, RandomForestClassifier):
        importances = st.session_state.model.feature_importances_
        feature_names = st.session_state.df.drop(columns=[st.session_state.y.name]).columns
        sorted_indices = importances.argsort()

        # Create a DataFrame for easier plotting
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names[sorted_indices],
            'Importance': importances[sorted_indices]
        })

        # Plotting Feature Importance
        st.write("Feature Importance Visualization:")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df, ax=ax, palette="viridis")
        ax.set_title("Feature Importance Ranking")
        st.pyplot(fig)

        # Display the DataFrame of feature importances for reference
        st.write("Feature Importance Data:")
        st.dataframe(feature_importance_df)
    else:
        st.warning("Please train a Random Forest model first.")

# 10. Download Results
if section == "Download Results":
    if 'model' in st.session_state:
        st.write("Download model or predictions as required.")
    else:
        st.warning("No model available to download.")
