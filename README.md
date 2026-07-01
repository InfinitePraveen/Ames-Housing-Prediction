# Ames Housing Price Prediction

## 🏠 Project Overview
This project aims to predict house sale prices using the Ames Housing dataset, which contains 79 explanatory variables describing various aspects of residential homes in Ames, Iowa. The dataset includes features ranging from basic (lot size, year built) to detailed (quality of materials, basement finish, garage condition).

### Dataset Characteristics
- **Total Records:** 2,930 observations
- **Features:** 79 explanatory variables
- **Target Variable:** SalePrice (continuous)
- **Data Source:** Ames, Iowa Assessor's Office

### Key Features
- **MS SubClass:** The building class
- **MS Zoning:** The general zoning classification
- **Lot Frontage:** Linear feet of street connected to property
- **Lot Area:** Lot size in square feet
- **Overall Qual:** Overall material and finish quality (1-10 scale)
- **Overall Cond:** Overall condition rating (1-10 scale)
- **Year Built:** Original construction date
- **Year Remod/Add:** Remodel date
- **SalePrice:** Sale price (target variable)

---

## 📂 Project Structure
ames-housing-prediction/
├── data/
│ ├── raw/
│ │ └── AmesHousing.csv # Original dataset (provided)
│ ├── processed/
│ │ └── ames_processed.csv # Cleaned and preprocessed data
│ └── external/ # For external data sources (if any)
│
├── notebooks/
│ ├── 01_EDA_and_Data_Cleaning.ipynb
│ ├── 02_Feature_Engineering.ipynb
│ ├── 03_Model_Development.ipynb
│ └── 04_Model_Evaluation_and_Interpretation.ipynb
│
├── src/
│ ├── init.py
│ ├── data/
│ │ ├── init.py
│ │ ├── load_data.py
│ │ ├── clean_data.py
│ │ └── preprocess.py
│ ├── features/
│ │ ├── init.py
│ │ ├── build_features.py
│ │ └── feature_selector.py
│ ├── models/
│ │ ├── init.py
│ │ ├── train_model.py
│ │ ├── predict.py
│ │ └── model_evaluation.py
│ └── visualization/
│ ├── init.py
│ ├── visualize.py
│ └── plots.py
│
├── models/
│ ├── best_model.pkl
│ ├── feature_importance.csv
│ └── model_metrics.json
│
├── web/ # For future deployment
│ ├── app.py
│ ├── templates/
│ │ └── index.html
│ └── static/
│ ├── css/
│ ├── js/
│ └── images/
│
├── tests/
│ ├── init.py
│ ├── test_data_loading.py
│ ├── test_preprocessing.py
│ └── test_models.py
│
├── config/
│ ├── config.yaml
│ └── logging.conf
│
├── reports/
│ ├── figures/
│ │ ├── feature_importance.png
│ │ ├── correlation_matrix.png
│ │ ├── price_distribution.png
│ │ └── residual_plots.png
│ └── final_report.md
│
├── requirements.txt
├── environment.yml
├── setup.py
├── Makefile
├── .gitignore
├── README.md
├── LICENSE
└── CHANGELOG.md