# Employee Salary Prediction

## Project Overview

This project focuses on building a machine learning model to predict an individual's income bracket (specifically, whether their annual income is `<=50K` or `>50K`). Understanding the factors influencing salary is crucial for fair compensation, career planning, and effective talent management.

## Problem Statement

In today's competitive job market, understanding factors that influence employee compensation is crucial for both organizations and individuals. Inconsistent or unfair salary structures can lead to decreased employee morale, high turnover rates, and difficulty in attracting top talent. For job seekers, having a realistic expectation of earnings based on their qualifications and experience is vital for career planning. This project aims to provide a data-driven approach to predict income brackets, moving beyond subjective evaluations.

## Dataset

The project utilizes the **Adult Income Dataset** (`adult 3.csv`), which contains various demographic and socio-economic features of individuals from a census database.

* **Source:** (If you downloaded it from a specific Kaggle competition or UCI ML repo, link it here. Otherwise, state it's a publicly available census dataset.)
* **Key Features:** Age, Workclass, Education, Marital Status, Occupation, Relationship, Race, Gender, Capital Gain, Capital Loss, Hours per Week, Native Country.
* **Target Variable:** `income` (categorized as `<=50K` or `>50K`).

## Algorithm & Development Process

Our predictive model is built using the **Random Forest Classifier**, a powerful ensemble learning algorithm.

* **Problem Type:** Binary Classification.
* **Data Preprocessing:** Involved handling missing values (replacing '?' with NaN and dropping rows), encoding the categorical target variable, and applying One-Hot Encoding to all other categorical features. A `scikit-learn Pipeline` was used for streamlined preprocessing.
* **Model Selection:** Random Forest was chosen for its high accuracy, robustness against overfitting, and ability to handle diverse feature types.
* **Imbalance Handling:** The `class_weight='balanced'` parameter was applied to the Random Forest model to address the class imbalance in the income categories, ensuring fair learning for both income groups.
* **Training & Evaluation:** The dataset was split into training and testing sets. The model was trained, and its performance was evaluated using accuracy, classification reports (precision, recall, F1-score), and confusion matrices. Feature importances were analyzed to identify key salary drivers.

## System & Library Requirements

To run this project, you'll need:

* **Operating Environment:** Windows, macOS, Linux, or a cloud platform like Google Colab.
* **Python Version:** Python 3.8 or higher.
* **Recommended RAM:** 8 GB or more.
* **Key Python Libraries:**
    * `pandas`
    * `numpy`
    * `scikit-learn`
    * `lightgbm` (even if not primary, it might be a dependency or was explored)
    * `joblib`
    * `matplotlib`
    * `seaborn`
    * `streamlit`
    * `pyngrok` (essential for running Streamlit apps in Colab/remote environments)

## How to Run the Project

Follow these steps to set up and run the Employee Salary Prediction application:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```
    (Replace `YourUsername` and `YourRepoName` with your actual GitHub details.)

2.  **Place the Dataset & Model:**
    * Ensure the `adult 3.csv` dataset is in the `data/` folder within the cloned repository.
    * Ensure the trained model file `salary_prediction_model_lgbm.pkl` is in the `models/` folder within the cloned repository.
    * *If you haven't already:* You need to run the model training script (e.g., `train_model.py` or your Colab notebook's training cells) to generate `salary_prediction_model_lgbm.pkl` and then upload it to the `models/` folder on GitHub.

3.  **Create Python Environment & Install Dependencies:**
    * It's highly recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    * Install all required libraries:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Run the Streamlit Application (Local):**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

5.  **Run the Streamlit Application (Google Colab):**
    * Open your Colab notebook.
    * **Upload `adult 3.csv` to `/content/data/` and `salary_prediction_model_lgbm.pkl` to `/content/models/`** (using the drag-and-drop method in the Colab file browser).
    * **Install Libraries:**
        ```python
        !pip install streamlit pyngrok lightgbm
        ```
    * **Set ngrok Authtoken:** (Get your token from [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken))
        ```python
        from pyngrok import ngrok
        ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN") # REPLACE THIS!
        ```
    * **Run the App:**
        ```python
        !nohup streamlit run app.py &
        import time
        time.sleep(4)
        from pyngrok import ngrok
        ngrok.kill()
        url = ngrok.connect(8501)
        print(f"Your Streamlit app is live at: {url}")
        ```
    * Click the generated URL to access the app.

## Results & Evaluation

The Random Forest Classifier achieved an accuracy of approximately **[Your Accuracy Here, e.g., 0.8524]** on the test set. The classification report provided insights into precision, recall, and F1-score for both income classes, highlighting the model's performance on the imbalanced dataset.

## Future Scope

* **Advanced Data Imputation:** Explore techniques like KNN Imputer for missing values.
* **Hyperparameter Tuning:** Conduct a thorough Grid Search or Randomized Search for Random Forest optimization.
* **Ensemble & Alternative Models:** Investigate stacking Random Forest with other strong classifiers (e.g., LightGBM, XGBoost) for potential accuracy gains.
* **Real-time Integration:** Develop the model for deployment within larger HR or recruitment systems.

---
