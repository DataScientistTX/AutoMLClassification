# AutoML Classification

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

AutoML Classification is a powerful, user-friendly tool for automated machine learning classification tasks. It streamlines the process of data preprocessing, model training, and evaluation, making it easier for data scientists and analysts to quickly develop and compare multiple classification models.

## Features

- **Automated Data Preprocessing**: Handles missing values, encoding, and feature scaling.
- **Multiple Classification Models**: Implements Logistic Regression, Random Forest, and Support Vector Machine (SVM) classifiers.
- **Model Evaluation and Comparison**: Provides accuracy scores and performance metrics for each model.
- **Feature Importance Visualization**: Helps identify the most influential features in your dataset.
- **Interactive Web Interface**: Built with Streamlit for an intuitive user experience.
- **Extensible Architecture**: Easily add new models or preprocessing steps.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/DataScientistTX/AutoMLClassification.git
   cd AutoMLClassification
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -e .
   ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app/main.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Upload your CSV file containing the dataset for classification, or use the provided example dataset.

4. Follow the on-screen instructions to preprocess your data, train models, and view results.

## Example Dataset

An example dataset (`example_dataset.csv`) is provided in the `examples/` directory. This dataset demonstrates the expected format and can be used to explore the application's functionality.

## Project Structure

```
AutoMLClassification/
│
├── src/                  # Source code for the project
│   ├── data_processing/  # Data loading and preprocessing
│   ├── feature_engineering/  # Feature creation and selection
│   ├── models/           # Model training and evaluation
│   └── visualization/    # Plotting and chart creation
│
├── app/                  # Streamlit application
│   └── main.py           # Main application file
│
├── tests/                # Unit tests
│
├── data/                 # Directory for user data (gitignored)
│
├── examples/             # Example datasets
│
├── requirements.txt      # Project dependencies
├── setup.py              # Package and distribution management
├── README.md             # Project documentation
└── .gitignore            # Git ignore file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://www.streamlit.io/) for the awesome web app framework
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [pandas](https://pandas.pydata.org/) for data manipulation and analysis

## Contact

Sercan Gul - [@DataScientistTX](https://x.com/DataScientistTX) - sercan.gul@gmail.com

Project Link: [https://github.com/DataScientistTX/AutoMLClassification](https://github.com/DataScientistTX/AutoMLClassification)