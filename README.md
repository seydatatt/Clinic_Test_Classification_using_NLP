# 🩺 Clinic Test Classification using NLP

This project analyzes written content in medical transcription data using natural language processing (NLP) techniques and classifies them according to medical specialties. The **Medical Transcriptions Dataset** provided on Kaggle was used.

## 📌 Project Objective
To predict the medical specialties to which the text content from medical transcriptions belongs.

## 🧰 Technologies and Libraries Used
**Python 3**

- pandas, numpy, seaborn, matplotlib  
- scikit-learn (TF-IDF, TruncatedSVD, Logistic Regression, Naive Bayes, Confusion Matrix)  
- NLTK (Tokenization, Lemmatization)  
- TSNE (for visualization)

## 📁 Dataset
**Source**: Medical Transcriptions Dataset - Kaggle

**Features**:

- `transcription`: Medical writing (text)  
- `medical_specialty`: Target class (specialty area)

The available data has been increased and filtered by selecting only the classes with at least 50 examples.

## 🔎 Applied Steps

### Data Cleaning:
- Removed missing values.
- Removed numbers, punctuation marks, and special characters.
- Applied lemmatization and lowercasing.

### Feature Extraction:
- Text data was converted to numerical vectors using **TfidfVectorizer**.
- Dimensionality reduction was applied with **TruncatedSVD** (used instead of PCA because TF-IDF is a sparse matrix).

### Data Visualization:
- Classes were visualized in 2D using **TSNE**.

### Model Training:
- Classification was performed using the **LogisticRegression** model.
- The data was split into training and test sets using `train_test_split`.

### Evaluation:
- Performance was measured using `classification_report` and `confusion_matrix`.

## 📊 Model Performance

The model works on a multi-class classification problem. Evaluation metrics used:

- Precision  
- Recall  
- F1-score

Additionally, the **Confusion Matrix** was visualized for a detailed performance analysis.


## 🚀 To Run

```bash
pip install -r requirements.txt
python clinic_test_classification.py
```

## 📌 Notes

- If **MultinomialNB** is to be used, the data must not contain negative values. Therefore, after the TF-IDF output is transformed using **TruncatedSVD**, checking for positive values is important.
- Since **PCA** does not work on sparse matrices, **TruncatedSVD** was preferred instead.

## 📄 License
- MIT License