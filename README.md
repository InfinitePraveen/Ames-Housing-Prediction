# Ames Housing Price Prediction

A machine learning project that predicts residential house sale prices using the Ames Housing dataset. The workflow covers data cleaning, feature engineering, model training, evaluation, and a simple web-based interface for prediction.

## Overview
The Ames Housing dataset contains detailed property information for homes sold in Ames, Iowa. Each record includes structural, neighborhood, and quality-related variables that influence the target variable, SalePrice. This project focuses on building a reliable regression pipeline that can handle both numerical and categorical features and produce interpretable results.

## Project Goals
- Predict house sale prices with strong regression performance
- Explore the dataset through exploratory data analysis
- Engineer meaningful features for better model accuracy
- Compare multiple machine learning models
- Provide a deployment-ready prediction interface

## Key Features
- Data loading, cleaning, and preprocessing pipelines
- Feature engineering and feature selection workflows
- Model training and evaluation using several regression algorithms
- Visualization of model performance and feature importance
- A lightweight Flask web app for predictions

## Dataset
- Source: Ames Housing dataset
- Target: SalePrice
- Type: Regression problem
- Size: 2,930 observations with 79 explanatory variables

## Project Structure
- data/: raw, processed, and external datasets
- notebooks/: end-to-end analysis notebooks
- src/: reusable Python modules for data processing, modeling, and visualization
- models/: saved model artifacts and evaluation outputs
- web/: Flask application and frontend assets
- tests/: unit tests for main project components
- reports/: generated reports and visualizations

## Setup Instructions
1. Clone the repository
2. Create and activate a virtual environment
3. Install the dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
- Run the notebooks in the notebooks/ directory for analysis and experimentation
- Train models using the scripts in src/models/
- Launch the web app with:

```bash
python web/app.py
```

## Model Workflow
1. Load and inspect the raw dataset
2. Clean missing values and standardize formats
3. Engineer and select useful features
4. Train multiple regression models
5. Evaluate model performance and interpret results

## Results and Output
The project generates:
- cleaned and processed datasets
- model metrics and comparison reports
- feature importance summaries
- visualization plots for analysis

## Testing
Run the test suite with:

```bash
pytest
```

## Connect With Me
I’m passionate about data science, machine learning, and building practical AI solutions.

- LinkedIn: https://www.linkedin.com/in/infinitepraveen/
- Replace the link above with your real LinkedIn profile URL to personalize this section.

## License
This project is intended for educational and portfolio purposes.
