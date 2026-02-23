# Modeling Injury Severity in NYC Traffic Accidents for Insurance Risk

This project builds a comprehensive machine-learning framework for predicting traffic accident injury severity in New York City[cite: 21]. [cite_start]It utilizes a robust data pipeline to integrate collision data with environmental factors, modeling crash outcomes as Non-Severe, Severe, and Fatal[cite: 22, 102]. [cite_start]The ultimate goal is to provide actionable insights for insurance industry pricing strategies and public safety initiatives like Vision Zero[cite: 18, 26].

### 📌 Objectives
* [cite_start]**Data Integration:** Clean and integrate NYC Motor Vehicle Collision data (Crashes, People, Vehicles) with hourly NOAA weather observations[cite: 99, 100, 101, 102].
* [cite_start]**Feature Engineering:** Transform free-form police text into standardized behavioral risk groups (e.g., Driver Distraction, Rule Violation, Under Influence)[cite: 180, 181, 184, 188].
* [cite_start]**Predictive Modeling:** Develop and evaluate multiple models, including an Ordered Logit baseline, Random Forest, XGBoost, and LightGBM, to predict severity class probabilities[cite: 357, 365, 367].
* [cite_start]**Interpretability:** Utilize SHAP (SHapley Additive exPlanations) to identify and quantify the key drivers of fatal and severe crash risks, such as rule violations, geographic location, and weather conditions[cite: 45, 476, 505, 506].

### 🧱 Tech Stack
* [cite_start]**Python** – For data preprocessing, feature engineering, and model training (LightGBM, XGBoost, Random Forest)[cite: 44, 367].
* **LaTeX** – For typesetting and formatting the final academic research paper.
* **Bash** – For executing and automating the workflow pipeline via shell scripts.
* **Git** – For version control.

### 📂 Folder Structure

The repository is organized as follows to separate source code, data, assets, and documentation:

* `Code/` – Scripts for data cleaning, modeling, and SHAP analysis.
* [cite_start]`Data/` – Storage directory for raw and processed datasets (NYC Collisions, NOAA Weather)[cite: 99, 102].
* [cite_start]`Figures/` – Generated visualizations, including SHAP plots, geographic risk maps, and distribution charts[cite: 131, 285, 435, 498].
* `Icon/` – Project icons and visual assets.
* `Paper/` – LaTeX source files for compiling the final research report.
* `References/` – Literature review materials, citations, and bibliography sources.
* [cite_start]`Tables/` – Output tables storing model performance metrics and ordered logit coefficients[cite: 396, 405].
* `DoWork.sh` – Shell script to automate the execution of the project pipeline.
* [cite_start]`_FPoojaPaper.pdf` – The final compiled academic paper detailing the study's methodology, economic theory framework, and insights[cite: 9, 69].
* `README.md` – Project documentation and setup guide.
