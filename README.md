# Employee Salary Prediction

## Project Overview

This project focuses on building a machine learning model to predict an individual's income bracket (specifically, whether their annual income is `<=50K` or `>50K`). Understanding the factors influencing salary is crucial for fair compensation, career planning, and effective talent management within various sectors.

## Problem Statement

In today's competitive job market, understanding factors that influence employee compensation is crucial for both organizations and individuals. Inconsistent or unfair salary structures can lead to decreased employee morale, high turnover rates, and difficulty in attracting top talent. For job seekers, having a realistic expectation of earnings based on their qualifications and experience is vital for career planning. This project aims to provide a data-driven approach to predict income brackets, moving beyond subjective evaluations and promoting more equitable salary assessments.

## Dataset

The project utilizes the **Adult Income Dataset** (`adult 3.csv`), which contains various demographic and socio-economic features of individuals extracted from a census database.

* **Source:** This dataset is widely available and derived from the 1994 US Census database.
* **Key Features:** Age, Workclass, Education, Marital Status, Occupation, Relationship, Race, Gender, Capital Gain, Capital Loss, Hours per Week, Native Country.
* **Target Variable:** `income` (categorized as `<=50K` or `>50K`).

## Algorithm & Development Process

The predictive model is built using the **Random Forest Classifier**, a robust ensemble machine learning algorithm.

* **Problem Type:** Binary Classification (predicting `<=50K` or `>50K` income).
* **Data Preprocessing:**
    * Handled missing values (represented as '?') by replacing them with `NaN` and then dropping incomplete rows.
    * The `income` target variable was converted into numerical labels using `LabelEncoder`.
    * Categorical features were transformed into numerical format using One-Hot Encoding.
    * A `scikit-learn Pipeline` was utilized for consistent and reproducible preprocessing.
* **Model Selection:** Random Forest was chosen for its high accuracy, robustness against overfitting, and ability to handle diverse feature types effectively.
* **Imbalance Handling:** The `class_weight='balanced'` parameter was applied to address the unequal distribution of income classes, preventing bias towards the majority class.
* **Training & Evaluation:** The dataset was split into training and testing sets. The Random Forest model was trained, and its performance was evaluated using accuracy, classification reports (precision, recall, F1-score), and confusion matrices. Feature importances were analyzed to identify key salary drivers.

## System & Library Requirements

To run this project, the following environment and dependencies are required:

* **Operating Environment:**
    * Windows, macOS, Linux, or a cloud-based environment (e.g., Google Colab)
    * Python 3.8 or higher
    * Minimum 8 GB RAM recommended for efficient data processing and model training.
* **Key Python Libraries:**
    * `pandas` (for data handling)
    * `numpy` (for numerical operations)
    * `scikit-learn` (for ML preprocessing and core functionalities)
    * `joblib` (for model saving/loading)
    * `matplotlib` (for visualizations)
    * `seaborn` (for enhanced visualizations)
    * `streamlit` (for the interactive web application)
    * `pyngrok` (essential for running Streamlit apps in Colab or remote environments)

## How to Run the Project

Follow these steps to set up and run the Employee Salary Prediction application:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/csb4658/employee-salary-prediction.git](https://github.com/csb4658/employee-salary-prediction.git)
    cd employee-salary-prediction
    ```

2.  **Ensure Data and Model Files are Present:**
    * Verify that `adult 3.csv` is located in the `data/` folder.
    * Verify that the trained model file `salary_prediction_model_lgbm.pkl` is in the `models/` folder.
    * *(If these files are not present after cloning, you will need to generate the model by running the training notebook/script and then place the dataset in the `data/` folder.)*

3.  **Set Up Python Environment & Install Dependencies:**
    * It is highly recommended to create and activate a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    * Install all required libraries using the provided `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Run the Streamlit Application (Local):**
    * If running on your local machine:
        ```bash
        streamlit run app.py
        ```
        This command will open the application in your default web browser.

5.  **Run the Streamlit Application (Google Colab):**
    * Open your `Employee Salary prediction.ipynb` notebook in Google Colab.
    * Ensure `app.py`, `adult 3.csv` (in `data/`), and `salary_prediction_model_lgbm.pkl` (in `models/`) are present in your Colab environment (e.g., uploaded via drag-and-drop or cloned from GitHub).
    * **Install Libraries in Colab:**
        ```python
        !pip install -r requirements.txt
        ```
        (or manually list them: `!pip install streamlit pyngrok lightgbm pandas scikit-learn numpy matplotlib seaborn joblib`)
    * **Set ngrok Authtoken:** (Obtain your authtoken from [ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken))
        ```python
        from pyngrok import ngrok
        ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN") # REPLACE "YOUR_NGROK_AUTH_TOKEN" with your actual token
        ```
    * **Execute the App in Colab:**
        ```python
        !nohup streamlit run app.py &
        import time
        time.sleep(4)
        from pyngrok import ngrok
        ngrok.kill()
        url = ngrok.connect(8501)
        print(f"Your Streamlit app is live at: {url}")
        ```
    * Click the generated `ngrok` URL to access the live application.

## Results & Evaluation

The Random Forest Classifier achieved an accuracy of approximately **[Your Accuracy Here, e.g., 0.8524]** on the test set. The classification report provided insights into precision, recall, and F1-score for both income classes, highlighting the model's performance on the dataset.

## Future Scope

To further enhance the Employee Salary Prediction project, future efforts could focus on:

* **Advanced Data Refinement:** Implement more sophisticated missing value imputation techniques (e.g., KNN Imputer) to maximize data utility.
* **Deeper Hyperparameter Tuning:** Rigorously optimize the Random Forest Classifier's parameters for peak performance.
* **Ensemble & Alternative Models:** Explore combining Random Forest with other strong classifiers (like LightGBM or XGBoost) or investigating deep learning approaches.
* **Real-time Integration:** Develop the model for deployment within larger HR or recruitment systems for real-time salary insights.

---
