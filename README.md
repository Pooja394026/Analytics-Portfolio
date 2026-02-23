# Modeling Injury Severity in NYC Traffic Accidents for Insurance Risk

This project builds a comprehensive machine-learning framework for predicting traffic accident injury severity in New York City. It utilizes a robust data pipeline to integrate collision data with environmental factors, modeling crash outcomes as Non-Severe, Severe, and Fatal. The ultimate goal is to provide actionable insights for insurance industry pricing strategies and public safety initiatives like Vision Zero.

### 📌 Objectives
* **Data Integration:** Clean and integrate NYC Motor Vehicle Collision data (Crashes, People, Vehicles) with hourly NOAA weather observations.
* **Feature Engineering:** Transform free-form police text into standardized behavioral risk groups (e.g., Driver Distraction, Rule Violation, Under Influence).
* **Predictive Modeling:** Develop and evaluate multiple models, including an Ordered Logit baseline, Random Forest, XGBoost, and LightGBM.
* **Interpretability:** Utilize SHAP (SHapley Additive exPlanations) to identify and quantify the key drivers of fatal and severe crash risks.

### 🧱 Tech Stack
* **Python** – Data preprocessing, feature engineering, and machine learning (LightGBM, XGBoost, Scikit-Learn).
* **LaTeX** – Typesetting and formatting for the final academic research paper.
* **Bash** – Workflow automation via the `DoWork.sh` shell script.
* **Git** – Version control and project management.

### 📂 Folder Structure

The repository is organized to separate source code, data, and documentation:

* **Code/** – Python scripts for data cleaning, modeling, and SHAP analysis.
* **Data/** – Raw and processed datasets (NYC Collisions, NOAA Weather).
* **Figures/** – Generated visualizations, including SHAP plots and geographic risk maps.
* **Icon/** – Project icons and visual assets.
* **Paper/** – LaTeX source files used to generate the final report.
* **References/** – Bibliography, citations, and literature review materials.
* **Tables/** – Output tables for model performance and logit coefficients.
* **DoWork.sh** – Main automation script to run the entire pipeline.
* **_FPoojaPaper.pdf** – The final academic research paper.

### 🚀 Getting Started
To replicate the results or run the pipeline:
1. Ensure Python 3.x and a Bash environment are installed.
2. Place the raw data files in the `Data/` folder.
3. Run the automation script:
   ```bash
   bash DoWork.sh
