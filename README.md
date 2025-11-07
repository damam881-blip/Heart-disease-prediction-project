# 🩺 Comprehensive Machine Learning Pipeline for Heart Disease Prediction

This project implements a full, end-to-end machine learning workflow to analyze, predict, and visualize heart disease risks using the Heart Disease UCI dataset.

---

## 🚀 Live Demo

You can view and interact with the live application deployed on Streamlit Community Cloud:

**[➡️ Click here to open the Live App](https://heart-disease-prediction-project-uq4dleibu2p7yaitrszdjg.streamlit.app/)**

---

##  workflow Project Workflow

This project follows a structured machine learning pipeline as defined by the provided sprint plan:

1.  **Data Preprocessing & Cleaning:**
    * Loaded the dataset from multiple sources (Cleveland, Hungary, etc.).
    * Handled significant missing values by dropping columns (`ca`, `slope`, `thal`).
    * Imputed remaining missing data using **Median** (for numerical features) and **Mode** (for categorical features).
    * Performed **One-Hot Encoding** for categorical features (`cp`, `restecg`) and **Binary Encoding** for (`sex`, `fbs`).
    * Created the final binary `target` variable (0 = No Disease, 1 = Disease).

2.  **Exploratory Data Analysis (EDA):**
    * Visualized the dataset to find patterns.
    * Identified `age`, `thalach`, and `cp_asymptomatic` as key indicators through a **Correlation Heatmap**.

3.  **Feature Engineering & Selection:**
    * **PCA (Principal Component Analysis):** Analyzed cumulative variance to find the optimal number of components.
    * **Feature Importance:** Used `RandomForestClassifier` to rank all 13 features.
    * **RFE (Recursive Feature Elimination):** Applied RFE to find the top 5 features.
    * **Chi-Square Test:** Used statistical testing to validate feature importance.

4.  **Model Training & Evaluation:**
    * Split the data (80% train, 20% test) using `stratify=y` to handle imbalance.
    * Trained and evaluated four different classification models:
        * Logistic Regression
        * Decision Tree
        * **Random Forest (Winner - 91.85% Base Accuracy)**
        * Support Vector Machine (SVM)
    * Plotted **ROC Curves** and compared **AUC scores** for all models.

5.  **Hyperparameter Tuning:**
    * Performed `RandomizedSearchCV` to find the best general parameters for the `RandomForest` model.
    * Performed a focused `GridSearchCV` to fine-tune the model, resulting in a more robust and less overfit model.

6.  **Model Export & Deployment:**
    * Saved the final optimized model (`final_model.pkl`) and the `StandardScaler` (`scaler.pkl`) using `joblib`.
    * Built an interactive web UI using **Streamlit**.
    * Deployed the application to **Streamlit Community Cloud** via GitHub.

---

## 🛠️ Tools & Technologies Used

* **Programming Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (`KMeans`, `PCA`, `RandomForestClassifier`, `SVC`, `LogisticRegression`, `GridSearchCV`)
* **Deployment:** Streamlit
* **Version Control:** Git & GitHub

---

## 📂 Project Structure
---

## 🏃 How to Run This Project Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/damam881-blip/Heart-disease-prediction-project.git](https://github.com/damam881-blip/Heart-disease-prediction-project.git)
    cd Heart-disease-prediction-project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run ui/app.py
    ```
