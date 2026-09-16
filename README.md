# Blood Cell Anomaly Detection 🩸

## Overview
This project is a complete End-to-End Machine Learning pipeline designed to detect anomalies in blood cells. It predicts whether a blood cell is normal or anomalous based on various medical features and laboratory readings.

## Technologies Used
* **Machine Learning Model:** XGBoost Classifier (achieved the highest accuracy among other tested models like Random Forest, Decision Tree, and Logistic Regression).
* **Backend:** FastAPI (serves the trained model via a REST API).
* **Frontend:** Streamlit (provides an interactive user interface for medical personnel to input cell data and get instant predictions).
* **Data Processing:** Pandas, Scikit-learn (handling One-Hot Encoding and data scaling).

## Project Structure
* `train.py`: Script for data preprocessing, model training, and exporting the final model.
* `main.py`: FastAPI server script.
* `app.py`: Streamlit frontend application.
* `xgb_model.pkl` & `model_columns.pkl`: The saved model and its feature columns.

## How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the FastAPI server: `uvicorn main:app --reload`
4. Run the Streamlit app: `streamlit run app.py`
