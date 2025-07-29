# -*- coding: utf-8 -*-

# Clinic_Test_Classification
#import libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import string 
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE 
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, classification_report

# Load Dataset 
data = pd.read_csv("mtsamples.csv")
data = data[data["transcription"].notna()]
data_categories = data.groupby(data["medical_specialty"])

c=1
for catg_name, data_category in data_categories:
    print(f"Category_{c}: {catg_name}: {len(data_categories)}")
    c = c+1
    
filtered_data_categories = data_categories.filter(lambda x: x.shape[0] > 50) 

final_data_categories = filtered_data_categories.groupby(filtered_data_categories["medical_specialty"])

c=1
for catg_name, data_category in data_categories:
    print(f"Category_{c}: {catg_name}: {len(data_categories)}")
    c = c+1

plt.figure()
sns.countplot(y="medical_specialty", data = filtered_data_categories, palette = "pastel" )
plt.show()

data_1 = filtered_data_categories[["transcription", "medical_specialty"]]
data_1 = data_1.drop(data_1[data_1["transcription"].isna()].index)

#Pre-Processing: Text Cleaning 
def clean_text(text):
    text = text.translate(str.maketrans("","",string.punctuation))
    text1 = "".join([w for w in text if not w.isdigit()]) 
    replace_by_space_re = re.compile("[/(){}\[\]\|@,;]")
    
    text2 = text1.lower()
    text2 = replace_by_space_re.sub("",text2)
    
    return text2

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    sentences = sent_tokenize(text)
    wordlist = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        wordlist.extend([lemmatizer.lemmatize(word) for word in words])
    return " ".join(wordlist)  

data_1["transcription"] = data_1["transcription"].apply(lemmatize_text)
data_1["transcription"] = data_1["transcription"].apply(clean_text)

# we will express the text to vector
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, min_df = 5, max_df=0.7)
tfidf = vectorizer.fit_transform(data_1["transcription"].tolist())
features_names = sorted(vectorizer.get_feature_names_out())

labels= data_1["medical_specialty"].tolist()
tsne = TSNE(n_components = 2)
tsne_result =tsne.fit_transform(tfidf.toarray())
sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], hue = labels)
plt.show()

# SVD Analize -> Model Training   

svd = TruncatedSVD(n_components=100, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf)
category_list = data.medical_specialty.unique()

X_train, X_test, y_train, y_test = train_test_split(tfidf_reduced, labels, random_state=42)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(classification_report(y_test, y_pred))

category_list = data_1["medical_specialty"].unique()
cm = confusion_matrix(y_test, y_pred, labels=category_list)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", xticklabels=category_list, yticklabels=category_list)
plt.title("Logistic Regression Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

















