import streamlit as st
import pandas as pd
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_processing import load_data
from src.feature_engineering import preprocess_data
from src.models import train_models, evaluate_models
from src.visualization import plot_feature_importance

def main():
    st.set_page_config(page_title="AutoML Classification", layout="wide")
    st.title("AutoML Classification")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())

        # Data preprocessing
        st.subheader("Data Preprocessing")
        df_processed = preprocess_data(df)
        st.write("Processed data shape:", df_processed.shape)

        # Feature importance
        st.subheader("Feature Importance")
        fig = plot_feature_importance(df_processed)
        st.pyplot(fig)

        # Model training
        st.subheader("Model Training")
        with st.spinner("Training models..."):
            models = train_models(df_processed)
        st.success("Models trained successfully!")

        # Model evaluation
        st.subheader("Model Evaluation")
        results = evaluate_models(models, df_processed)
        st.write(results)

        # Best model
        best_model = results.iloc[0]['Model']
        st.success(f"Best performing model: {best_model}")

        # Download results
        csv = results.to_csv(index=False)
        st.download_button(
            label="Download model evaluation results",
            data=csv,
            file_name="model_evaluation_results.csv",
            mime="text/csv",
        )

    else:
        st.info("Please upload a CSV file to get started.")

    # Add information about the example dataset
    st.sidebar.markdown("---")
    st.sidebar.subheader("Example Dataset")
    st.sidebar.info(
        "An example dataset is provided in the `examples/` directory. "
        "You can use this to explore the functionality of the app."
    )

    # Add project information
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This app demonstrates automated machine learning "
        "for classification tasks. Upload your dataset and "
        "let the app handle preprocessing, model training, "
        "and evaluation."
    )
    st.sidebar.markdown(
        "[View on GitHub](https://github.com/DataScientistTX/AutoMLClassification)"
    )

if __name__ == "__main__":
    main()