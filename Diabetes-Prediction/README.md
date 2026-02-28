
# Disease Prediction System (Machine Learning)

This project is a **multi-class disease prediction system** built using Machine Learning.
It predicts possible diseases based on given symptoms using multiple classifiers and a final ensemble (majority voting) approach.

The goal of this project is to explore:

* Handling imbalanced medical datasets
* Training multiple ML models
* Comparing performance using cross-validation
* Building a simple symptom-based prediction function

---

## 📌 Project Overview

In this project, we:

1. Load and preprocess a disease dataset
2. Encode categorical labels
3. Handle class imbalance using **Random Oversampling**
4. Train multiple ML models:

   * Decision Tree
   * Random Forest
   * Support Vector Machine (SVM)
   * Naive Bayes
5. Evaluate models using:

   * 5-Fold Stratified Cross Validation
   * Accuracy Score
   * Confusion Matrix
6. Combine predictions using **Majority Voting (Ensemble Method)**
7. Create a custom function to predict disease from user input symptoms

---

## 🛠️ Technologies & Libraries Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (RandomOverSampler)

---

## ⚙️ Data Preprocessing

* Loaded dataset using `pandas`
* Encoded disease labels using `LabelEncoder`
* Separated features (X) and target (y)
* Visualized class distribution
* Applied **RandomOverSampler** to handle class imbalance
* Filled missing values with 0
* Encoded categorical features (e.g., gender if present)

---

## 🤖 Models Used

### 1️⃣ Decision Tree

Simple tree-based classifier.

**Mean Accuracy (CV):** ~53.9%

---

### 2️⃣ Random Forest

Ensemble of decision trees.

**Mean Accuracy (CV):** ~54.2%
**Training Accuracy:** 68.98%

---

### 3️⃣ Support Vector Machine (SVM)

**Mean Accuracy (CV):** ~50%
**Training Accuracy:** 60.53%

---

### 4️⃣ Naive Bayes

**Training Accuracy:** 37.98%

---

## 🧠 Final Ensemble Model (Majority Voting)

We combined predictions from:

* Random Forest
* Naive Bayes
* SVM

Final prediction is chosen using **mode (majority vote)**.

**Combined Model Accuracy:** 60.64%

---

## 📊 Model Evaluation

Each model was evaluated using:

* Stratified 5-Fold Cross Validation
* Accuracy Score
* Confusion Matrix Visualization

Confusion matrices were plotted using Seaborn heatmaps for better visualization.

---

## 🔮 Custom Disease Prediction Function

You can predict disease by passing symptoms as a comma-separated string.

### Example:

```python
predict_disease("skin_rash,fever,headache")
```

### Example Output:

```python
{
  'Random Forest Prediction': 'Peptic ulcer disease',
  'Naive Bayes Prediction': 'Impetigo',
  'SVM Prediction': 'Peptic ulcer disease',
  'Final Prediction': 'Peptic ulcer disease'
}
```

---

## 🚀 How It Works

1. User enters symptoms
2. System converts symptoms into binary feature vector
3. Each model predicts disease
4. Final result is selected using majority voting

Just tell me what you need, Aakash 🚀

